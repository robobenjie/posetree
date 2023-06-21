# posetree

`posetree` is an object-oriented, thread-safe, programmer-first, Python library for dealing with transforms, poses and frames, designed especially for robotics.

## Installation

```
pip install posetree
```

## Features

* All poses are aware of all the frames which allows for powerful operations with simple code.
* Write pose math designed for humans to read.
* Think about quaternions as little as possible.
* Never wonder if you are supposed to pre-multiply or post-multiply, or if you forgot an inverse somewhere.
* Standalone and easy to integrate into any robotics stack.

## Basic Usage

```python
from scipy.spatial.transform import Rotation
from posetree import Pose

looking_at = Pose.from_position_and_rotation([0, 0, 1], Rotation.identity(), "camera", pose_tree)

height = looking_at.in_frame("robot").z

base_target = base_pose.translate([2, 0, 0]).rotate_about_z(np.pi/4)

base_target2 = base_pose.point_x_at(human_pose).translate([1, 0, 0])

x, y, _ = base_target.in_frame("map").position
theta = base_target.angle_about_z_to(map_origin)
navigate_to(x, y, theta)
```

## Philosophy of Transforms, Poses and Frames

`posetree` takes an opinionated stance on some words that other robotics stacks use somewhat interchangably. 

- **Transform**: This comes from the verb, to transform. It is how you get from one place to another. It is an operation to change your location from one to another.
    - Example: “Take 10 steps forward and then turn 90 degrees to your left”
    - Example: "Translate 1m in x, and 2m in y then rotate 30 degrees about the y axis."

- **Pose**: This is a noun. It is a physical location and orientation in 3D space.
    - Example: “Standing at my front door facing the street.”
    - Example: "The location of the left camera lens, z pointing out."

Notice we can take our example pose (standing at my front door facing the street) and transform it into another pose by using our transform instructions (take 10 steps forward and then turn 90 degrees to your left). If we do that we have a new pose that is a bit in front of my house and (in my case) facing our detached garage.

- **Frame**: This is a pose that is so important we’ve decided to give it a name. 
    - Example: We could name “Standing at my front door facing the street” “journey_start” and then we could describe that other pose by saying, “from journey_start take 10 steps forward and turn 90 degrees to your left”
    - Example: We could name the left camera pose `"camera"`.

You can sequence Transforms by multiplying them together, (as is traditional). This has a semantic meaning: "do this operation and then this other operation." However you cannot sequentially apply positions in 3D space so there is no multiply operator for Pose.

## Anchoring Poses in Frames.

When constucting poses it is useful to think of what you expect the pose to be fixed relative to. For example, you might
detect an apple on a table and get a pose from your perception system in the `camera` frame, but the apple is fixed relative to the table, so you will
want to store your pose object with a the parent frame equal to `odometry/map/world`. Then, even if the robot moves, `apple_pose` will still refer to the
best estimation of the apple's true location (with the usual caviats about localization drift and noise).

```python
apple_pose = Pose(camera_t_apple, "camera", pose_tree).in_frame("world")
```

Likewise if you have a bin on the back of a mobile robot and you want to define a drop-objects-into-bin pose right above it you can store
that in the `robot` frame.

```python
drop_pose = Pose.from_position_and_quaternion([-.3, 0.1, 0.25], [0, 0, 0, 1]), "robot", pose_tree)
```

When designing motion APIs with this library, you should be liberal in what frames you accept, and internally convert them to the frame you want to work in. 

For example, in a function to move the arm to a pose:
```python
def move_to_pose(self, target_pose: Pose):
    # Convert the target to be relative to the base of the robot so we can execute the motion.
    target_pose = target_pose.in_frame("robot")

    # Best Practice: turn Poses into Transforms at the last moment before acting on them.
    arm_motion_primitives.move_arm_to_pose_relative_to_base(target_pose.transform)
```

This formulation lets you combine the perception outputs and the motion methods for things like this:

```python
def grasp_apple(self, apple_pose: Pose):
    # pregrasp is a pose 15 cm above the apple, with z pointing at it.
    pregrasp = apple_pose.translate([0, 0, 0.15], frame="world").point_z_at(apple_pose)
    self.move_to_pose(pregrasp)

    # Move down in the local 'z' of pregrasp until we touch the apple.
    self.move_to_pose_until_contact(pregrasp.translate([0, 0, 0.2]))

    # If we feel a contact too early, raise some reasonable error:
    # We can check the distance using distance_to, even though the frames are different.
    if apple_pose.distance_to(get_tool_pose()) > 0.1:
        return "Whoops, we probably didn't get the apple."

    # Close the gripper and move up a bit from where ever we are, to lift the apple.
    self.close_gripper()
    self.move_to_pose(get_tool_pose().translate([0, 0, -0.1]))

    # Drop it in the bin.
    self.move_to_pose(drop_pose)
    self.open_gripper()
```

## Poses are Immutable

Poses are immutable, and methods like `translate` return a new pose object. Immutability is nice because it makes them safe to pass into methods and also thread safe, but there is a gotcha to watch out for:

```python
# BAD!!! Do Not Do!
p1.translate([1,0,0])
p1.rotate_about_z(np.rad2deg(90))
p1.with_position_x(5)
# Surprise! p1 has not changed!

# Better
p1 = p1.translate([1,0,0])
p1 = p1.rotate_about_z(np.rad2deg(90))
p1 = p1.with_position_x(5)
# We are replacing p1 with the new pose returned, so this works

# My favorite
p2 = p1.translate([1,0,0]).rotate_about_z(np.rad2deg(90)).with_position_x(5)
```

## Immutability and Moving Frames

While a Pose is immutable, the parent frame can (and does!) change over time relative to other frames, meaning that an individual pose can move relative
to other frames. (For example a pose defined in the "robot" frame will conceptually move as the robot moves relative to a 'world'
frame, even though its position and orientation remain immutably constant.)

To make this very concrete, say the robot starts out at the world origin:

```python
pose_in_robot_frame = pose.from_position_and_rotation([1,2,3], Rotation.identity(), "robot", pose_tree)
pose_in_world_frame = pose_in_robot_frame.in_frame("world")

pose_in_robot_frame.position # [1,2,3]
pose_in_world_frame.position # [1,2,3]

# Now the robot moves 1 meter forward in the world frame.
robot.drive_forward_in_x(1)

# Poses are immutable so this has not changed. One pose is [1,2,3] from the robot frame origin, one is [1,2,3] from the world origin.
pose_in_robot_frame.position # [1,2,3]
pose_in_world_frame.position # [2,2,3]

# But if we express the one in robot frame in the world frame, we see that it is now 1 meter forward in x.
pose_in_robot_frame.in_frame("world").position # [2,2,3]
```

## Connecting it to the rest of your stack.

To connect a pose_tree instance you need to subclass PoseTree. Lets say you have an (fictional) object called `MyTransformManager` in your stack that subscribes to pose messages and implements `get_transform`. You would write something like:

```python
class MyPoseTree(CustomFramePoseTree):
    """My implementation of PoseTree to integrate with MyTransformManager"""

    def __init__(self, transform_manager: MyTransfrormManager):
        self._tfm = transformManager

    def _get_transform(self, parent_frame: str, child_frame: str, timestamp: Optional[float] = None) -> Transform:
        transform_data = self._tfm.get_transform(parent_frame, child_frame, timestamp)
        return Transform.from_position_and_quaternion(
            [transform_data.tx, transform_data.ty, transform_data.tz],
            [transform_data.qx, transform_data.qy, transform_data.qz, transform_data.qw]
        )
```

Then you can use it like this:

```python
my_tf_manager = MyTransformManager()
my_tf_manager.subscribe(Channels.ALL_ROBOT_CHANNELS) # idk I'm making this API up.
pose_tree = MyPoseTree(my_tf_manager)
p1 = Pose.from_position_and_rotation([1,2,3], Rotation.identity(), "robot", pose_tree)
p2 = p1.in_frame("camera", timestamp = robot.get_current_timestamp() - 1.0)
```

## Documentation

For more detailed information about the API and how to use PoseTree, check out the [documentation](https://htmlpreview.github.io/?https://raw.githubusercontent.com/robobenjie/posetree/main/docs/posetree/pose.html).

## Contributing

We welcome contributions! This is my first open source project so I don't know what I'm doing, but I'd especially like it if folks wanted to create PoseTree object implementations to wrap tf2 nicely for ROS and ROS2.

## License

PoseTree is licensed under the [MIT license](https://github.com/robobenjie/posetree/blob/main/LICENSE).