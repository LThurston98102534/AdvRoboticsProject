#include <atomic>
#include <mutex>
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


#include "sensor_msgs/image_encodings.h"

namespace enc = sensor_msgs::image_encodings;

const int eight_point_check_connect = 2;
const int unknown_lower_thresh = 110;
const int unknown_upper_thresh = 160;
const int free_thresh = 255;

const cv::Vec3b white = cv::Vec3b(255,255,255);
const cv::Vec3b red = cv::Vec3b(0,0,255);
const cv::Vec3b blue = cv::Vec3b(255, 0 , 0);
const cv::Vec3b grey = cv::Vec3b(127, 127, 127);
const cv::Vec3b green = cv::Vec3b(0,255,0);
const cv::Vec3b black = cv::Vec3b(0, 0, 0);

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

namespace frontier_explorer
{
class FrontierExplorer
{
public:
  // Constructor
  explicit FrontierExplorer(ros::NodeHandle& nh);

  // Publich methods
  void mainLoop();

private:
  // Variables
  nav_msgs::OccupancyGrid map_{};
  cv::Mat map_image_{};


  int image_msg_count_ = 0;

  // Transform listener
  tf2_ros::Buffer transform_buffer_{};
  tf2_ros::TransformListener transform_listener_{ transform_buffer_ };


  // Velocity command publisher
  ros::Publisher cmd_vel_pub_{};

  ros::Subscriber map_sub_{};


  // Image transport and subscriber
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_{};


  // Action client
  actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> move_base_action_client_{ "move_base", true };

  // Private methods
  geometry_msgs::Pose2D getPose2d();
  void imageCallback(const sensor_msgs::ImageConstPtr& image_msg_ptr);
  void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);

  image_transport::Publisher image_pub_;

  cv_bridge::CvImage searchImageForFrontiers(cv::Mat& image, cv_bridge::CvImage& colour_image);

  geometry_msgs::Pose convertPixelToGlobal(cv::Point& pixel);

  double calculateDistanceRobottoPoint(geometry_msgs::Pose& global_point);

  geometry_msgs::Pose2D calculateMoveBasePosition(geometry_msgs::Pose2D& robot_pose, geometry_msgs::Pose& goal_pose);


  bool map_parameters_set_;
  unsigned int map_size_in_cells_x_;
  unsigned int map_size_in_cells_y_;
  double map_resolution_;
  double map_origin_x_;
  double map_origin_y_;
  std::string global_frame_;
  cv::Point map_centre_;

  geometry_msgs::Pose goal_pose_global_;
  std::mutex global_goal_pose_mutex_;
  cv::Point goal_pose_pixel_;

};



// Constructor
FrontierExplorer::FrontierExplorer(ros::NodeHandle& nh) : it_{ nh }, map_parameters_set_(0), map_size_in_cells_x_(0), map_size_in_cells_y_(0), map_resolution_(0.0), map_origin_x_(0.0), map_origin_y_(0.0)
{
  
  map_sub_ = nh.subscribe("/map", 1, &FrontierExplorer::mapCallback, this);

  // Subscribe to the map
  image_sub_ = it_.subscribe("map_image/full", 1, &FrontierExplorer::imageCallback, this);

  // Advertise "cmd_vel" publisher to control TurtleBot manually
  cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1, false);


  // Action client for "move_base"
  ROS_INFO("Waiting for \"move_base\" action...");
  move_base_action_client_.waitForServer();
  ROS_INFO("\"move_base\" action available");
/*
  // Reinitialise AMCL
  ros::ServiceClient global_localization_service_client = nh.serviceClient<std_srvs::Empty>("global_localization");
  std_srvs::Empty srv{};
  global_localization_service_client.call(srv);
*/
  image_pub_ = it_.advertise("/frontier_exploration", 1);
}

geometry_msgs::Pose2D FrontierExplorer::getPose2d()
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




void FrontierExplorer::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    if(!map_parameters_set_){
	ROS_INFO("Map Callback");
	global_frame_ = msg->header.frame_id;

	map_size_in_cells_x_ = msg->info.width;
	map_size_in_cells_y_ = msg->info.height;
	map_resolution_ = msg->info.resolution;
	map_origin_x_ = msg->info.origin.position.x;
	map_origin_y_ = msg->info.origin.position.y;

	map_centre_.x = ((-1*map_origin_x_)/map_resolution_);
	map_centre_.y = map_size_in_cells_y_-((-1*map_origin_y_)/map_resolution_);

	map_parameters_set_ = true;

	std::cout << "Global Frame: " << global_frame_ << std::endl;
	std::cout << "Size in Cells X: " << map_size_in_cells_x_ << std::endl;
	std::cout << "Size in Cells Y: " << map_size_in_cells_y_ << std::endl;
	std::cout << "Resolution: " << map_resolution_ << std::endl;
	std::cout << "Origin X: " << map_origin_x_ << std::endl;
	std::cout << "Origin Y: " << map_origin_y_ << std::endl;

        map_sub_.shutdown();
    }

    
}



void FrontierExplorer::imageCallback(const sensor_msgs::ImageConstPtr& image_msg_ptr)
{
  cv_bridge::CvImagePtr image_ptr;

  try
    {
      if (enc::isColor(image_msg_ptr->encoding)) {
          image_ptr = cv_bridge::toCvCopy(image_msg_ptr, enc::BGR8);
      
      } else {
          image_ptr = cv_bridge::toCvCopy(image_msg_ptr, enc::MONO8);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }


  // This is the OpenCV image
  cv::Mat& image = image_ptr->image;
  cv_bridge::CvImage colour_image;

  cv::cvtColor(image, colour_image.image, CV_GRAY2RGB);

  std::cout << "Searching for Frontiers" << std::endl;

  if(map_parameters_set_){
      colour_image = searchImageForFrontiers(image, colour_image);
  }

  // Create CV Image for displaying output of blob detection
  cv_bridge::CvImage frontier_image;
  frontier_image.encoding = "bgr8";
  frontier_image.header = std_msgs::Header();
  frontier_image.image = colour_image.image;

  // Publish image to topic
  image_pub_.publish(frontier_image.toImageMsg());

  //ROS_INFO("Image Callback");

}


cv_bridge::CvImage FrontierExplorer::searchImageForFrontiers(cv::Mat& image, cv_bridge::CvImage& colour_image){
    
     // Declare a very high initial minimum value (it should be replaced)
    double min_distance = (std::pow(std::pow(image.rows, 2) + std::pow(image.cols,2),0.5));

    // Iterate through every pixel in grayscale image
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            unsigned char &pixel = image.at<unsigned char>(i, j);

            // If pixel is white
            if (pixel == free_thresh) {

                // Iterate through all connected cells
                for(int k = -1; k < eight_point_check_connect; k ++){
                    for(int l = -1; l < eight_point_check_connect; l ++){

                        // If one connected cell is grey, white cell is a frontier
                        if((image.at<unsigned char>(i+k, j+l) > unknown_lower_thresh) && (image.at<unsigned char>(i+k, j+l) < unknown_upper_thresh)){
                            cv::Point frontier(j, i);

                            // Break loop
                            k = eight_point_check_connect;
                            l = eight_point_check_connect;

                            // Colour frontier pixel blue in RGB image
                            colour_image.image.at<cv::Vec3b>(frontier.y, frontier.x) = blue;

                            // Iterate through all connected cells
                            for(int m = -1; m < eight_point_check_connect; m ++){
                                for(int n = -1; n < eight_point_check_connect; n ++){

                                    // If one connected cell to the frontier is white, it is a potential goal pose
                                    if((image.at<unsigned char>((frontier.y + m), (frontier.x + n)) == free_thresh)){

                                        // Initialise bool value
                                        bool valid_point = true;

                                        // Check potential goal pose pixel, if it is connected to a grey cell, it is a frontier, but if no grey cells, it is a potential goal pose
                                        for(int o = -1; o < eight_point_check_connect; o ++){
                                            for(int p = -1; p < eight_point_check_connect; p ++){
                                                if(((image.at<unsigned char>((frontier.y + m + o), (frontier.x + n + p)) > unknown_lower_thresh) && (image.at<unsigned char>((frontier.y + m + o), (frontier.x + n + p)) < unknown_upper_thresh)) || ((image.at<unsigned char>((frontier.y + m + o), (frontier.x + n + p)) < unknown_lower_thresh)) || ((image.at<unsigned char>((frontier.y + m + 2*o), (frontier.x + n + 2*p)) < unknown_lower_thresh))){
                                                    valid_point = false;
                                                }
                                            }

                                        }

                                        if(valid_point == true){
                                            cv::Point frontier_pixel((frontier.x + n), (frontier.y + m));

                                            // Convert from pixel to local reference frame
                                            geometry_msgs::Pose check_point = convertPixelToGlobal(frontier_pixel);

                                            // Calculate distance from potential goal pose to robot
                                            double d = calculateDistanceRobottoPoint(check_point);

                                            // If calculated distance is less than the minimum distance calculated to any potential goal pose, this point is closer and hence should be considered the goal pose
                                            if(d < min_distance){

                                                // Update new distance
                                                min_distance = d;

                                                // Create unique lock to safely access member variables
                                                std::unique_lock<std::mutex> goal_lock(global_goal_pose_mutex_);

                                                // Update local goal pose member variables
                                                goal_pose_global_.position.x = check_point.position.x;
                                                goal_pose_global_.position.y = check_point.position.y;

						goal_pose_pixel_ = frontier_pixel;

                                                goal_lock.unlock();
                                            }
                                        }

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    colour_image.image.at<cv::Vec3b>(goal_pose_pixel_.y, goal_pose_pixel_.x) = green;  

    colour_image.image.at<cv::Vec3b>(map_centre_.y, map_centre_.x) = red;  

    return colour_image;
}


double FrontierExplorer::calculateDistanceRobottoPoint(geometry_msgs::Pose& global_point)
{
    geometry_msgs::Pose2D robot_pose = getPose2d();

    double distance = std::pow(std::pow((global_point.position.x - robot_pose.x),2)+std::pow((global_point.position.y - robot_pose.y),2),0.5);

    return distance;

}


geometry_msgs::Pose FrontierExplorer::convertPixelToGlobal(cv::Point& pixel){

    geometry_msgs::Pose global_point;
    global_point.position.x = (pixel.x - map_centre_.x)*map_resolution_;
    global_point.position.y = (-1*(pixel.y - map_centre_.y))*map_resolution_;

    return global_point;

}

geometry_msgs::Pose2D FrontierExplorer::calculateMoveBasePosition(geometry_msgs::Pose2D& robot_pose, geometry_msgs::Pose& goal_pose){

    geometry_msgs::Pose2D move_base_pose;

    move_base_pose.x = goal_pose.position.x - robot_pose.x;
    move_base_pose.y = goal_pose.position.y - robot_pose.y;

    return move_base_pose;

}


void FrontierExplorer::mainLoop()
{
  
  // Stop turning
  geometry_msgs::Twist twist{};
  twist.angular.z = 0.;
  cmd_vel_pub_.publish(twist);


  // This loop repeats until ROS shuts down, you probably want to put all your code in here
  while (ros::ok())
  {
    ROS_INFO("mainLoop");
    std::cout << "Goal Pose Pixel: (" << goal_pose_pixel_.x << ", " << goal_pose_pixel_.y << ")" << std::endl;
    std::cout << "Goal Pose: (" << goal_pose_global_.position.x << ", " << goal_pose_global_.position.y << ")" << std::endl;

    geometry_msgs::Pose2D robot_pose = getPose2d();

    //geometry_msgs::Pose2D move_base_position = calculateMoveBasePosition(robot_pose, goal_pose_global_);
    geometry_msgs::Pose2D move_base_position;
    move_base_position.x = goal_pose_global_.position.x;
    move_base_position.y = goal_pose_global_.position.y;

    std::cout << "Move Base Pose: (" << move_base_position.x << ", " << move_base_position.y << ")" << std::endl;

    // Send a goal to "move_base" with "move_base_action_client_"
    move_base_msgs::MoveBaseActionGoal action_goal{};

    action_goal.goal.target_pose.header.frame_id = "map";
    action_goal.goal.target_pose.pose = pose2dToPose(move_base_position);

    ROS_INFO("Sending goal...");
    move_base_action_client_.sendGoal(action_goal.goal);

    // Delay so the loop doesn't run too fast
    ros::Duration(1.0).sleep();
  }
}

}  // namespace frontier_explorer

int main(int argc, char** argv)
{
  ros::init(argc, argv, "frontier_explorer");

  ros::NodeHandle nh{};

  frontier_explorer::FrontierExplorer fe(nh);

  // Asynchronous spinner doesn't block
  ros::AsyncSpinner spinner(1);
  spinner.start();

  fe.mainLoop();

  return 0;
}
