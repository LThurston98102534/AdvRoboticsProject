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
  // Use this method to identify when the brick is visible
  brick_found_ = 0;

  // The camera publishes at 30 fps, it's probably a good idea to analyse images at a lower rate than that
  if (image_msg_count_ < 15)
  {
    image_msg_count_++;
    return;
  }
  else
  {
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
    cv::minEnclosingCircle(contours[i], c, r);

    // Only save larger blobs (red brick). If a larger blob is seen, it is the brick and set the brick found variable to 1
    if(r > 75) {
        std::cout << "Circle at: " << c.x << ", " << c.y << " with radius: " << r << std::endl;
        std::cout << std::endl;

        centre.push_back(c);
        radius.push_back(r);
        brick_found_ = 1;
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
    if(brick_found_ < 1){
	// Autonomously drive around the maze
        // Check current position of robot
        // Calculate next position to drive to
        // Set action goal
 	// Send goal to move_base


    } else {
	// Calculate relative position of brick from TurtleBot
        // Calculate next drive instruction to drive towards the brick
        // Set action goal
 	// Send goal to move_base
        // Set flag upon reaching the brick

    }

    // Get the state of the goal
    actionlib::SimpleClientGoalState state = move_base_action_client_.getState();

    if (state == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      // Stop TurtleBot and await next instruction


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
