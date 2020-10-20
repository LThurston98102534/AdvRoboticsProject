#include <atomic>
#include <cmath>

#include <opencv2/core.hpp>

#include <ros/ros.h>
#include <nav_msgs/GetMap.h>
#include <tf2_ros/transform_listener.h>
#include <std_srvs/Empty.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <mutex>
#include <stdlib.h>
#include <time.h>

// Constants used for driving at brick
const int stopping_dist_from_brick = 400;
const int brick_angular_thresh = 120;

namespace
{
double wrapAngle(double angle)
{
  // Function to wrap an angle between 0 and 2*Pi
  while (angle < 0.)
  {
    angle += 2 * M_PI;
  }

  while (angle > (2 * M_PI))
  {
    angle -= 2 * M_PI;
  }

  return angle;
}

// Function used to return the 2D pose of the robot
geometry_msgs::Pose pose2dToPose(const geometry_msgs::Pose2D& pose_2d)
{
  geometry_msgs::Pose pose{};

  pose.position.x = pose_2d.x;
  pose.position.y = pose_2d.y;

  pose.orientation.w = std::cos(pose_2d.theta);
  pose.orientation.z = std::sin(pose_2d.theta / 2.);

  return pose;
}
}  // namespace

namespace brick_search
{
class BrickSearch
{
public:
  // Constructor
  explicit BrickSearch(ros::NodeHandle& nh);

  // Publich methods
  void mainLoop();

private:
  // Variables
  nav_msgs::OccupancyGrid map_{};
  cv::Mat map_image_{};
  std::atomic<bool> localised_{ false };
  std::atomic<bool> brick_found_{ false };
  int image_msg_count_ = 0;

  // Centre of the map
  cv::Point map_centre_;

  // List of goal poses to drive between
  std::vector<geometry_msgs::Pose> ValidGoalList;
  
  
  // Transform listener
  tf2_ros::Buffer transform_buffer_{};
  tf2_ros::TransformListener transform_listener_{ transform_buffer_ };

  // Subscribe to the AMCL pose to get covariance
  ros::Subscriber amcl_pose_sub_{};

  // Velocity command publisher
  ros::Publisher cmd_vel_pub_{};

  // Image transport and subscriber
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_{};

  // Action client
  actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> move_base_action_client_{ "move_base", true };

  // Private methods
  geometry_msgs::Pose2D getPose2d();
  void amclPoseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose_msg);
  void imageCallback(const sensor_msgs::ImageConstPtr& image_msg_ptr);
  
  // Function used to divide map into a grid and set valid Goal Poses
  void AssessMap();
  // Function used to sort the goal poses in the list in descending order based on distance from the robot
  void orderGoalPoses(geometry_msgs::Pose2D &robot_pose);
  
  image_transport::Publisher image_pub_;
  image_transport::Publisher image_map_pub_;

  // Properties used to drive at the red brick
  cv::Point2f brick_centre_;
  float brick_circle_radius_;
  std::mutex brick_properties_mutex_;

  // Properties of the map used to find valid goal poses
  int img_width_;
  int img_height_;
  int img_props_set_;
  std::mutex img_properties_mutex_;

  // Flag used to indicate the robot has reached the target location at the brick
  std::atomic<bool> navigated_to_brick_{ false};

  // Service client used to clear the costmaps to allow robot to plan valid paths
  ros::ServiceClient clear_costmaps_client_;

};

// Constructor
BrickSearch::BrickSearch(ros::NodeHandle& nh) : it_{ nh }
{
  // Wait for "static_map" service to be available
  ROS_INFO("Waiting for \"static_map\" service...");
  ros::service::waitForService("static_map");

  // Get the map
  nav_msgs::GetMap get_map{};

  if (!ros::service::call("static_map", get_map))
  {
    ROS_ERROR("Unable to get map");
    ros::shutdown();
  }
  else
  {
    map_ = get_map.response.map;
    ROS_INFO("Map received");
  }

  // Set up clear cost maps client
  clear_costmaps_client_ = nh.serviceClient<std_srvs::Empty>("/move_base/clear_costmaps");

  // Calculate the map centre
  map_centre_.x = ((-1*map_.info.origin.position.x)/map_.info.resolution);
  map_centre_.y = ((-1*map_.info.origin.position.y)/map_.info.resolution);
  
  // This allows you to access the map data as an OpenCV image
  map_image_ = cv::Mat(map_.info.height, map_.info.width, CV_8U, &map_.data.front());

  
  // Wait for the transform to be become available
  ROS_INFO("Waiting for transform from \"map\" to \"base_link\"");
  while (ros::ok() && !transform_buffer_.canTransform("map", "base_link", ros::Time(0.)))
  {
    ros::Duration(0.1).sleep();
  }
  ROS_INFO("Transform available");

  // Subscribe to "amcl_pose" to get pose covariance
  amcl_pose_sub_ = nh.subscribe("amcl_pose", 1, &BrickSearch::amclPoseCallback, this);

  // Subscribe to the camera
  image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &BrickSearch::imageCallback, this);

  // Advertise "cmd_vel" publisher to control TurtleBot manually
  cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1, false);

  // Action client for "move_base"
  ROS_INFO("Waiting for \"move_base\" action...");
  move_base_action_client_.waitForServer();
  ROS_INFO("\"move_base\" action available");

  // Reinitialise AMCL
  ros::ServiceClient global_localization_service_client = nh.serviceClient<std_srvs::Empty>("global_localization");
  std_srvs::Empty srv{};
  global_localization_service_client.call(srv);

  // Set up image publishers
  image_pub_ = it_.advertise("/blob_detection", 1);
  image_map_pub_ = it_.advertise("/map_image", 1);

  // Initialise image properties
  img_props_set_ = 0;
  img_width_ = 0;
  img_height_ = 0;

  // Divide map into grid and collate goal poses into a list
  AssessMap();
  
}

geometry_msgs::Pose2D BrickSearch::getPose2d()
{
  // Lookup latest transform
  geometry_msgs::TransformStamped transform_stamped =
      transform_buffer_.lookupTransform("map", "base_link", ros::Time(0.), ros::Duration(0.2));

  // Return a Pose2D message
  geometry_msgs::Pose2D pose{};
  pose.x = transform_stamped.transform.translation.x;
  pose.y = transform_stamped.transform.translation.y;

  double qw = transform_stamped.transform.rotation.w;
  double qz = transform_stamped.transform.rotation.z;

  pose.theta = qz >= 0. ? wrapAngle(2. * std::acos(qw)) : wrapAngle(-2. * std::acos(qw));

  return pose;
}

void BrickSearch::amclPoseCallback(const geometry_msgs::PoseWithCovarianceStamped& pose_msg)
{
  // Check the covariance
  double frobenius_norm = 0.;

  for (const auto e : pose_msg.pose.covariance)
  {
    frobenius_norm += std::pow(e, 2.);
  }

  frobenius_norm = std::sqrt(frobenius_norm);

  if (frobenius_norm < 0.05)
  {
    localised_ = true;

    // Unsubscribe from "amcl_pose" because we should only need to localise once at start up
    amcl_pose_sub_.shutdown();
  }
}

void BrickSearch::imageCallback(const sensor_msgs::ImageConstPtr& image_msg_ptr)
{
  ROS_INFO("imageCallback");

  // Initialise brick found flag to be false
  bool brick_flag = false;

  // The camera publishes at 30 fps, it's probably a good idea to analyse images at a lower rate than that
  if (image_msg_count_ < 5)
  {
    image_msg_count_++;
    return;
  }
  else
  {
    // Use this method to identify when the brick is visible
    image_msg_count_ = 0;
  }

  // Copy the image message to a cv_bridge image pointer
  cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(image_msg_ptr);

  // This is the OpenCV image
  cv::Mat& image = image_ptr->image;
  
  // Convert the camera image into a HSV image
  cv::Mat hsv_image;
  cv::cvtColor(image, hsv_image, CV_RGB2HSV);

  // Check that the image properties have already been set. Only run once to prevent unnecessary running of the code
  if (img_props_set_ < 1){
    // Store values in the appropriate member variables protected by a mutex
    std::unique_lock<std::mutex> img_properties_lock(img_properties_mutex_);
    img_width_ = hsv_image.cols;
    img_height_ = hsv_image.rows;
    
    img_properties_lock.unlock();

    // Increment flag so the code will only run once
    img_props_set_++;
  }

  // Set boundary limits for detecting red objects (STILL NEED TO BE REFINED)
  cv::Scalar min(0, 200, 100);
  cv::Scalar max(15/2, 255, 255);

  // Generate boundary image set detected red components to white and the rest black
  cv::Mat threshold_mat;
  cv::inRange(hsv_image, min, max, threshold_mat);
  
  
  // Initialise variables ready for blob/object detection
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<cv::Point2i> centre;
  std::vector<int> radius;

  // Perform blob detection
  cv::findContours(threshold_mat.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

  // Iterate through each of the contours/blobs detected and create a Circle surrounding the blob
  int count = contours.size();
  for(int i=0; i < count; i++){
    cv::Point2f c;
    float r;
    float max_radius = 0.0;
    cv::minEnclosingCircle(contours[i], c, r);

    // Only save larger blobs (red brick). If a larger blob is seen, it is the brick and set the brick found variable to 1
    if(r > 200) {
        centre.push_back(c);
        radius.push_back(r);

        // Only store the radius and centre properties of the largest blob detected in the image (this should be the brick)
        if(max_radius < r){
	    max_radius = r;
	    std::unique_lock<std::mutex> brick_properties_lock(brick_properties_mutex_);
            brick_circle_radius_ = r;
	    brick_centre_ = c;

            brick_properties_lock.unlock();
	}
        // Set temporary brick found flag to true
        brick_flag = true;
    }
  }

  // Set Member variable flag to indicate to the main loop that the brick has been found
  brick_found_ = brick_flag;

  // Print red Circle onto the image for visualisation purposes
  count = centre.size();
  cv::Scalar red(255,0,0);  
  for(int i = 0; i < count; i++){
    cv::circle(threshold_mat, centre[i], radius[i], red, 3);
  }


  // Create CV Image for displaying output of blob detection
  cv_bridge::CvImage threshold_image;
  threshold_image.encoding = "mono8";
  threshold_image.header = std_msgs::Header();
  threshold_image.image = threshold_mat;

  // Publish image to topic
  image_pub_.publish(threshold_image.toImageMsg());

  ROS_INFO_STREAM("brick_found_: " << brick_found_);
}

// Function used to divide the map into a grid and store valid goal poses in the list
void BrickSearch::AssessMap() {
  ROS_INFO_STREAM("Assessing the Map!");

  // Initialise variables
  int MapWidth = map_image_.cols;
  int MapHeight = map_image_.rows;
  const int RegionSize = 9; // Set grid size to be 9 px x 9 px
  const int PIXEL_BLACK = 0;
  bool validGoal;
  
  double HorizontalRegions = MapWidth / RegionSize;  //Divide the map into smaller squares
  double VerticalRegions = MapHeight / RegionSize;   //Divide the map into smaller squares
    
  for (int i = 0; i < HorizontalRegions; i++) {
    for (int j = 0; j < VerticalRegions; j++) {
      int px = (i*RegionSize + (RegionSize/2)); //Calculate the centre  pixel of each square.
      int py = (j*RegionSize + (RegionSize/2)); //Calculate the centre  pixel of each square.
      unsigned char &pixel = map_image_.at<unsigned char>(px,py); //Get the colour of the pixel.
      
      if (pixel == PIXEL_BLACK) { //If black we want to investigate this point
        //Set it to be a potentially valid goal pose
        validGoal = true;

        // Check that within 5 pixels left, right, up and down, it is all free space to provide enough room for the robot to drive to and turn around the goal pose
        for(int k = -5; k < 6; k ++){
          for(int l = -5; l < 6; l ++){
            // If the pixel is not black, it is either a wall or unknown space
            if((map_image_.at<unsigned char>(px+k, py+l) > PIXEL_BLACK)) {
               // Reset validGoal flag
               validGoal = false;
               // Break out of for loop
               k = 6;
               l = 6;
            }
          }
        }

        // If it is a valid Goal position, store it in the ValidGoalList vector
        if(validGoal){
          geometry_msgs::Pose RobotPose;
          RobotPose.position.y = (px - map_centre_.x)*map_.info.resolution; //Convert the Pixel to Robot Pose.
          RobotPose.position.x = ((py - map_centre_.y))*map_.info.resolution; //Convert the Pixel to Robot Pose.
          RobotPose.orientation.w = 0; //Set default value for angle reference to 0. 
          
          ROS_INFO_STREAM("GRID X: " << RobotPose.position.x << ", GRID Y: " << RobotPose.position.y << " Added to ValidGoalList");
          ValidGoalList.push_back(RobotPose); //Store the Robot Pose for later.
        }     

      }  
    }
  }
  
  ROS_INFO_STREAM("List has " << ValidGoalList.size() << " Items");
  
}

// Function used to sort the goal poses in the ValidGoalList vector in descending order based on distance away from the robot
void BrickSearch::orderGoalPoses(geometry_msgs::Pose2D &robot_pose){
  // Initialise temporary variables used in function
  double distance_1;
  double distance_2;
  geometry_msgs::Pose temp_pose;
  
  // Sort in descending order based on distance from the robot
  for ( int i = 0; i < (int)ValidGoalList.size(); ++ i ) {
    for ( int j = 0; j < (int)ValidGoalList.size()-1; ++ j ) {

      // Calculate the distance between the ith vector element and the (i+1)th vector element and the robot
      distance_1 = std::pow(std::pow((ValidGoalList.at(j).position.x - robot_pose.x),2)+std::pow((ValidGoalList.at(j).position.y - robot_pose.y),2),0.5);
      distance_2 = std::pow(std::pow((ValidGoalList.at(j+1).position.x - robot_pose.x),2)+std::pow((ValidGoalList.at(j+1).position.y - robot_pose.y),2),0.5);

      // If the current element is closer than the next, swap them in the order of the vector
      if ( distance_1 < distance_2 ) {
        temp_pose = ValidGoalList[j];
        ValidGoalList[j] = ValidGoalList[j+1];
        ValidGoalList[j+1] = temp_pose;
      }
    }
  }
}



void BrickSearch::mainLoop()
{
  // Wait for the TurtleBot to localise  
  ROS_INFO("Localising...");
  while (ros::ok())
  {
    // Turn slowly
    geometry_msgs::Twist twist{};
    twist.angular.z = 0.5;
    cmd_vel_pub_.publish(twist);

    if (localised_)
    {
      ROS_INFO("Localised");
      break;
    }

    ros::Duration(0.1).sleep();
  }

  // Clear costmaps to provide a clean map for the robot to drive between
  std_srvs::Empty srv;
  clear_costmaps_client_.call(srv);

  // Pause to allow the service to execute
  ros::Duration(1.0).sleep();

// UNCOMMENT BELOW LINES TO PUBLISH MAP IMAGE USED TO DIVIDE MAP INTO GRID AND CALCULATE GOAL POSITIONS
/*
  // Create CV Image for displaying output of map 
  cv_bridge::CvImage map_image;
  map_image.encoding = "mono8";
  map_image.header = std_msgs::Header();
  map_image.image = map_image_;


  // Publish image to topic
  image_map_pub_.publish(map_image.toImageMsg());
*/

  // Obtain current robot pose and re order the Valid Goal List vector based on current pose
  geometry_msgs::Pose2D robot_pose = getPose2d();
  orderGoalPoses(robot_pose);
  
  // Stop turning
  geometry_msgs::Twist twist{};
  twist.angular.z = 0.;
  cmd_vel_pub_.publish(twist);

  // Initialise variables used in main loop
  int goal_offset = 0;
  int goal_set = 0;
 
  // Main loop which repeats until ROS shuts down
  while (ros::ok())
  {
    ROS_INFO("mainLoop");
    // If brick has not yet been found, calculate next goal pose and drive using move base
    if(!brick_found_){
      // If goal has not yet been set, run code, otherwise wait for goal status to be published or brick to be found
      if(goal_set == 0){
        // Initialise variable for goal position
        geometry_msgs::Pose2D move_goal_position;
        
        if(ValidGoalList.size() >= 3) {
          // Initialise random number generator seed
          srand (time(NULL));
          // Select random goal position from the first two thirds of the list (two third of poses furthest away from robot)
          goal_offset = rand() % (int)(2*(ValidGoalList.size()/3));
        } else {
          goal_offset = 0;
        }
       
        // Set target position and store in move_goal_position
        move_goal_position.x = ValidGoalList.at(goal_offset).position.x;
        move_goal_position.y = ValidGoalList.at(goal_offset).position.y;
        
        // If robot is located in top half of map, point it down, if it is in the bottom half of the map, point it up, this should keep the robot driving forward most of the time
        if (move_goal_position.x < 0) {
          move_goal_position.theta = 0;
                  
        }else{
          move_goal_position.theta = 1.57;
          
        }
        
        // Initialise action client message
        move_base_msgs::MoveBaseActionGoal action_goal{};

        // Set up action client
        action_goal.goal.target_pose.header.frame_id = "map";
        action_goal.goal.target_pose.pose = pose2dToPose(move_goal_position);

        ROS_INFO_STREAM("Send goal... X: " << move_goal_position.x << ", Y: " << move_goal_position.y << ", Theta: " << move_goal_position.theta);

        // Send goal to move base
        move_base_action_client_.sendGoal(action_goal.goal);
        
        // Set flag so the goal is only set once
        goal_set = 1;

      }

    } else {
        // If the brick has been found in the image, cancel all current Move Base goals
        move_base_action_client_.cancelAllGoals();
        goal_set = 0;
        ROS_INFO_STREAM("Brick Found! Navigating to brick!");

        // Initialise temporary variables used in this function
        double brick_radius;
        double diff_x;
	int img_width_for_calc;

	// Calculate relative position of brick in the Turtlebot's camera frame
	std::unique_lock<std::mutex> brick_properties_lock(brick_properties_mutex_);
        brick_radius = brick_circle_radius_;
	diff_x = brick_centre_.x - (img_width_/2);
        img_width_for_calc = img_width_;

        brick_properties_lock.unlock();
	geometry_msgs::Twist twist{};
	
	ROS_INFO_STREAM("Brick Enclosing Circle Radius: " << brick_radius);
        
        // If not close enough to the brick, adjust to drive towards it
	if(brick_radius < stopping_dist_from_brick){
	    
            // Centre brick in the FOV of the camera
	    if(std::abs(diff_x) > brick_angular_thresh){
		ROS_INFO_STREAM("Twisting to face brick");

                // Stop moving linearly
		twist.linear.x = 0.;
                // Rotate at a speed proportional to the distance of the centre of the blob from the centre of the camera's FOV
  	        twist.angular.z = -1*(diff_x/img_width_for_calc)*0.3;

            // Once centred, drive towards the brick
	    } else {
		ROS_INFO_STREAM("Driving at brick");
                
                // Stop turning
		twist.angular.z = 0.;
                // Drive straight at brick at a speed proportional to the distance of the brick from the robot
		twist.linear.x = 0.01 + ((stopping_dist_from_brick - brick_radius)/stopping_dist_from_brick)*0.2;

	    }

        } else {
            // Once reaching the brick, stop moving and set flag to true
  	    twist.angular.z = 0.;
	    twist.linear.x = 0.;

	    navigated_to_brick_ = true;

        }

	ROS_INFO_STREAM("Twist Linear: " << twist.linear.x << ", " << twist.linear.y << ", " << twist.linear.z);
        ROS_INFO_STREAM("Twist Angular: " << twist.angular.x << ", " << twist.angular.y << ", " << twist.angular.z);
       
        // Publish linear and angular velocities of the robot to the cmd_vel topic to drive the Turtlebot
        cmd_vel_pub_.publish(twist);

    }

    // Get the state of the goal
    actionlib::SimpleClientGoalState state = move_base_action_client_.getState();
    ROS_INFO_STREAM(state.getText());
    
    // If a path cannot be found and the goal is aborted by Move Base, set a new goal
    if(state==actionlib::SimpleClientGoalState::ABORTED){
       ROS_INFO_STREAM("Goal Aborted!!");
       
       // Reset goal_set flag so a new goal will be generated
       goal_set = 0;
    }

    // If either the robot has successfully reached the goal or has finished it's movement at the brick
    if ((state == actionlib::SimpleClientGoalState::SUCCEEDED) || (navigated_to_brick_ == true))
    {
      // Check if flag is set for reaching the brick, if so, shutdown ROS
      if(navigated_to_brick_ == true){
        ROS_INFO_STREAM("Found the Brick and reached the Target Location. Shutting Down ROS");

        // Shutdown ROS
        ros::shutdown();
      } else {
        // Print the state of the goal
        ROS_INFO_STREAM(state.getText());

        // Reset goal_set flag so a new goal will be generated
        goal_set = 0;

        // Remove goal position from the ValidGoalList
        for(int i = goal_offset; i < (ValidGoalList.size() - 1); i++){
	  ValidGoalList.at(i) = ValidGoalList.at(i+1);
        }

        ValidGoalList.pop_back();

        // Obtain robot's pose and re order the ValidGoalList vector
        geometry_msgs::Pose2D robot_pose = getPose2d();
        orderGoalPoses(robot_pose);
        

        // If all waypoints have been hit and the brick has not been found, reset waypoints and start exploring map again
        if ((ValidGoalList.size() == 0)) {
          ROS_INFO_STREAM("Entire Map Explored! Restarting Exploration");
          AssessMap();
          geometry_msgs::Pose2D robot_pose = getPose2d();
          orderGoalPoses(robot_pose);
        }
      }
      

    }

    // Delay so the loop doesn't run too fast
    ros::Duration(0.5).sleep();
  }
}

}  // namespace brick_search

int main(int argc, char** argv)
{
  ros::init(argc, argv, "brick_search");

  ros::NodeHandle nh{};

  brick_search::BrickSearch bs(nh);

  // Asynchronous spinner doesn't block
  ros::AsyncSpinner spinner(1);
  spinner.start();

  bs.mainLoop();

  return 0;
}
