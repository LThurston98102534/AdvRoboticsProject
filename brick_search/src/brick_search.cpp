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

const int stopping_dist_from_brick = 400;
const int brick_angular_thresh = 150;

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

  image_transport::Publisher image_pub_;


  cv::Point2f brick_centre_;
  float brick_circle_radius_;
  std::mutex brick_properties_mutex_;

  int img_width_;
  int img_height_;
  int img_props_set_;
  std::mutex img_properties_mutex_;

  std::atomic<bool> navigated_to_brick_{ false};

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

  image_pub_ = it_.advertise("/blob_detection", 1);
  //image_pub_ = it_.advertise("/map_image", 1);

  img_props_set_ = 0;
  img_width_ = 0;
  img_height_ = 0;

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
  // The camera publishes at 30 fps, it's probably a good idea to analyse images at a lower rate than that
  if (image_msg_count_ < 15)
  {
    image_msg_count_++;
    return;
  }
  else
  {
    // Use this method to identify when the brick is visible
    brick_found_ = false;
    image_msg_count_ = 0;
  }

  // Copy the image message to a cv_bridge image pointer
  cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(image_msg_ptr);

  // This is the OpenCV image
  cv::Mat& image = image_ptr->image;

  // You can set "brick_found_" to true to signal to "mainLoop" that you have found a brick
  // You may want to communicate more information
  // Since the "imageCallback" and "mainLoop" methods can run at the same time you should protect any shared variables
  // with a mutex
  // "brick_found_" doesn't need a mutex because it's an atomic
  
  // Convert the camera image into a HSV image
  cv::Mat hsv_image;
  cv::cvtColor(image, hsv_image, CV_RGB2HSV);

  if (img_props_set_ < 1){
    std::unique_lock<std::mutex> img_properties_lock(img_properties_mutex_);
    img_width_ = hsv_image.cols;
    img_height_ = hsv_image.rows;
    
    img_properties_lock.unlock();

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
    if(r > 75) {
        std::cout << "Circle at: " << c.x << ", " << c.y << " with radius: " << r << std::endl;
        std::cout << std::endl;

        centre.push_back(c);
        radius.push_back(r);

        if(max_radius < r){
	    max_radius = r;
	    std::unique_lock<std::mutex> brick_properties_lock(brick_properties_mutex_);
            brick_circle_radius_ = r;
	    brick_centre_ = c;

            brick_properties_lock.unlock();
	}
        brick_found_ = true;
    }
  }

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

  ROS_INFO("imageCallback");
  ROS_INFO_STREAM("brick_found_: " << brick_found_);
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


    // Create CV Image for displaying output of blob detection
    /*
    cv_bridge::CvImage threshold_image;
    threshold_image.encoding = "mono8";
    threshold_image.header = std_msgs::Header();
    threshold_image.image = map_image_;
    image_pub_.publish(threshold_image.toImageMsg());

    std::cout << "Map Width: " << map_.info.width;
    std::cout << "Map Height: " << map_.info.height;
    std::cout << "Map Resolution: " << map_.info.resolution;
    std::cout << "Map Origin: " << map_.info.origin.position.x << ", " << map_.info.origin.position.y;
*/
    ros::Duration(0.1).sleep();
  }

  // Stop turning
  geometry_msgs::Twist twist{};
  twist.angular.z = 0.;
  cmd_vel_pub_.publish(twist);

  // The map is stored in "map_"
  // You will probably need the data stored in "map_.info"
  // You can also access the map data as an OpenCV image with "map_image_"

  ROS_INFO_STREAM("map_ resolution: " << map_.info.resolution);


  // Here's an example of getting the current pose and sending a goal to "move_base":
  geometry_msgs::Pose2D pose_2d = getPose2d();

  ROS_INFO_STREAM("Current pose: " << pose_2d);
/*
  // Move forward 0.5 m
  pose_2d.x += 0.5 * std::cos(pose_2d.theta);
  pose_2d.y += 0.5 * std::sin(pose_2d.theta);

  ROS_INFO_STREAM("Target pose: " << pose_2d);

  // Send a goal to "move_base" with "move_base_action_client_"
  move_base_msgs::MoveBaseActionGoal action_goal{};

  action_goal.goal.target_pose.header.frame_id = "map";
  action_goal.goal.target_pose.pose = pose2dToPose(pose_2d);

  ROS_INFO("Sending goal...");
  move_base_action_client_.sendGoal(action_goal.goal);
*/

  // This loop repeats until ROS shuts down, you probably want to put all your code in here
  while (ros::ok())
  {
    ROS_INFO("mainLoop");
    if(!brick_found_){

	std::cout << "Brick not found" << std::endl;
	// Autonomously drive around the maze
        // Check current position of robot
        // Calculate next position to drive to
        // Set action goal
 	// Send goal to move_base


    } else {
        std::cout << "Brick Found! Navigating to brick" << std::endl;
        double brick_radius;
        double diff_x;
	int img_width_for_calc;

	// Calculate relative position of brick from TurtleBot
	std::unique_lock<std::mutex> brick_properties_lock(brick_properties_mutex_);
        brick_radius = brick_circle_radius_;
	diff_x = brick_centre_.x - (img_width_/2);
        img_width_for_calc = img_width_;

        brick_properties_lock.unlock();
	geometry_msgs::Twist twist{};
	
	std::cout << "Brick Radius: " << brick_radius << std::endl;

	if(brick_radius < stopping_dist_from_brick){
	    
	    if(std::abs(diff_x) > brick_angular_thresh){
		std::cout << "Twisting to face brick" << std::endl;
		twist.linear.x = 0.;
  	        twist.angular.z = -1*(diff_x/img_width_for_calc)*0.3;

	    } else {
		std::cout << "Driving at brick!" << std::endl;
		twist.angular.z = 0.;
		twist.linear.x = ((stopping_dist_from_brick - brick_radius)/stopping_dist_from_brick)*0.2;

	    }

        } else {
  	    twist.angular.z = 0.;
	    twist.linear.x = 0.;

	    navigated_to_brick_ = true;

        }

	std::cout << "Twist Linear: " << twist.linear.x << ", " << twist.linear.y << ", " << twist.linear.z << std::endl;
        std::cout << "Twist Angular: " << twist.angular.x << ", " << twist.angular.y << ", " << twist.angular.z << std::endl;

        cmd_vel_pub_.publish(twist);

    }

    // Get the state of the goal
    actionlib::SimpleClientGoalState state = move_base_action_client_.getState();

    if (state == actionlib::SimpleClientGoalState::SUCCEEDED)
    {

      // Print the state of the goal
      ROS_INFO_STREAM(state.getText());

      // Check if flag is set of reaching the brick, if so, shutdown ROS
      // Shutdown when done
      //ros::shutdown();
    }

    // Delay so the loop doesn't run too fast
    ros::Duration(0.2).sleep();
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
