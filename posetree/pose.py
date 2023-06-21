"""
`posetree` is a module for representing and manipulating poses in 3D space. There are two main classes: `Pose` and `PoseTree`.

A `Pose` is an immutable location and orientation in 3D space, relative to a frame (i.e. 1 meter forward in X from the robot base).

A `PoseTree` is a object that keeps track of frames as they change over time, so that poses can know how to transform themselves into other frames, 
and do math operations with poses in different frames at particular points in time. You will need to subclass PoseTree to hook it up to your robot's
odometry system.

"""


import numpy as np
from typing import Optional, Tuple, Union, Sequence
from scipy.spatial.transform import Rotation, Slerp
from abc import ABC, abstractmethod
from contextlib import contextmanager
import uuid

    
class Pose(object):
    """
    A Pose is an immutable location and orientation in 3D space.
    A pose is defined with a `parent_frame` (where to start) and a transform (how to get to the pose from the parent frame).

    Note: While a Pose is immutable, the parent frame can (and does!) change over time relative to other frames, meaning that an individual pose can move relative
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
    Of course, the above example assumes that the `pose_tree` is correctly hooked up to the robot odometry system, so the poses know that the frames 
    have moved.

    When constucting poses it is useful to think of what you expect the pose to be fixed relative to. For example, you might
    detect an apple on a table and get a pose from your perception system in the `camera` frame, but the apple is fixed relative to the table, so you will
    want to store your pose object with a the parent frame equal to `odometry/map/world`. Then, even if the robot moves, apple_pose will still refer to the
    best estimation of the apple's true location (with the usual caviats about localization drift and noise).

    ```
    apple_pose = Pose(camera_t_apple, "camera", pose_tree).in_frame("world")
    ```

    Likewise if you have a bin on the back of a mobile robot and you want to define a drop-objects-into-bin pose right above it you can store
    that in the `robot` frame.

    ```
    drop_pose = Pose.from_position_and_quaternion([-.3, 0.1, 0.25], [0, 0, 0, 1]), "robot", pose_tree)
    ```

    When designing motion APIs with this library, you should be liberal in what frames you accept, and internally convert them to the frame you want to work in. 

    For example, in a function to move the arm to a pose:
    ```
    def move_to_pose(self, target_pose: Pose):
        # Convert the target to be relative to the base of the robot so we can execute the motion.
        target_pose = target_pose.in_frame("robot")

        # Best Practice: turn Poses into Transforms at the last moment before acting on them.
        arm_motion_primitives.move_arm_to_pose_relative_to_base(target_pose.transform)
    ```

    This formulation lets you combine the perception outputs and the motion methods for things like this:

    ```
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

    """
    def __init__(self, transform: "Transform", parent_frame: str, pose_tree: "PoseTree") -> None:
        """Create a pose from a transform, frame, and pose tree.
        
        Args:
            transform: The transform from the frame to the pose.
            parent_frame: The frame that the transform is relative to.
            pose_tree: The world that the pose exists in.
        """
        self._transform = transform
        self._frame = parent_frame
        self._pose_tree = pose_tree

    @classmethod
    def from_position_and_quaternion(cls, position: Sequence[float], quaternion: Sequence[float], parent_frame: str, pose_tree: "PoseTree") -> "Pose":
        """Create a pose from a position and quaternion.
        
        Args:
            position: The position of the transform.
            quaternion: The quaternion of the transform, in xyzw order.
            parent_frame: The frame that the transform is relative to.
            pose_tree: The world that the pose exists in.
        
        Returns:
            A new Pose.
        """
        return cls(Transform.from_position_and_quaternion(position, quaternion), parent_frame, pose_tree)
    
    @classmethod
    def from_position_and_rotation(cls, position: Sequence[float], rotation: Rotation, parent_frame: str, pose_tree: "PoseTree") -> "Pose":
        """Create a transform from a position and rotatoin.
        
        Args:
            position: The position of the transform.
            rotation: The rotation of the transform.
            parent_frame: The frame that the transform is relative to.
            pose_tree: The world that the pose exists in.
        
        Returns:
            A new Pose.
        """
        return cls(Transform(position, rotation), parent_frame, pose_tree)

    @property
    def transform(self) -> "Transform":
        """The transform from the parent frame to the pose."""
        return self._transform
    
    @property
    def position(self) -> np.ndarray:
        """The position of the pose relative to the parent frame."""
        return self.transform.position
    
    @property
    def rotation(self) -> Rotation:
        """The rotation of the pose relative to the parent frame."""
        return self.transform.rotation
    
    @property
    def frame(self) -> str:
        """The parent frame of the pose."""
        return self._frame
    
    @property
    def pose_tree(self) -> "PoseTree":
        """The PoseTree object containing the world that the pose exists in."""
        return self._pose_tree
    
    @property
    def x(self) -> float:
        """The x coordinate of the position."""
        return self.transform.x
    
    @property
    def y(self) -> float:
        """The y coordinate of the position."""
        return self.transform.y
    
    @property
    def z(self) -> float:
        """The z coordinate of the position."""
        return self.transform.z
    
    @property
    def x_axis(self) -> np.ndarray:
        """The x axis of the pose."""
        return self.transform.x_axis
    
    @property
    def y_axis(self) -> np.ndarray:
        """The y axis of the pose."""
        return self.transform.y_axis
    
    @property
    def z_axis(self) -> np.ndarray:
        """The z axis of the pose."""
        return self.transform.z_axis
    
    def in_frame(self, parent_frame: str, timestamp: Optional[float] = None) -> "Pose":
        """Return a new pose representing the same location in space expressed in a different frame.
        
        Args:
            new_parent_frame: The new frame that the transform is relative to.
        
        Returns:
            A new pose with the same transform but in a different frame.
        """
        new_transform = self.pose_tree.get_transform(parent_frame, self.frame, timestamp) * self.transform
        return Pose(new_transform, parent_frame, self.pose_tree)
    
    def with_position(self, position: Sequence[float]) -> "Pose":
        """Return a new pose with the same rotation but a different location relative to the parent frame.
        
        Args:
            position: The new location of the pose.
        
        Returns:
            A new pose with the same rotation but a different location.
        """
        new_transform = Transform(position, self.rotation)
        return Pose(new_transform, self.frame, self.pose_tree)
    
    def with_rotation(self, rotation: Rotation) -> "Pose":
        """Return a new pose with the same location but a different rotation.
        
        Args:
            rotation: The new rotation of the pose.
        
        Returns:
            A new pose with the same location but a different rotation.
        """
        new_transform = Transform(self.position, rotation)
        return Pose(new_transform, self.frame, self.pose_tree)
    
    def with_rotation_matching(self, other: "Pose") -> "Pose":
        """Return a new pose with the same position but a different orientation, matching another pose.

        This method is frame aware, so even if the other pose has a different frame, the rotation will be matched correctly.
        
        Args:
            other: The pose to match the rotation of.
        
        Returns:
            A new pose with the same position but a different orientation.
        """
        return self.with_rotation(other.in_frame(self.frame).rotation)
    
    def with_position_x(self, x: float, *, frame: Optional[str] = None) -> "Pose":
        """Return a new pose with the same y and z coordinates but a different x coordinate.
        
        Args:
            x: The new x coordinate of the pose.
            frame: The frame that the x coordinate is relative to. If None, the x coordinate is relative to the pose's frame.
        
        Returns:
            A new pose with the same y and z coordinates but a different x coordinate.
        """
        if frame is None:
            return self.with_position((x, self.y, self.z))
        else:
            in_frame = self.in_frame(frame)
            return in_frame.with_position((x, in_frame.y, in_frame.z)).in_frame(self.frame)
        
    def with_position_y(self, y: float, *, frame: Optional[str] = None) -> "Pose":
        """Return a new pose with the same x and z coordinates but a different y coordinate.
        
        Args:
            y: The new y coordinate of the pose.
            frame: The frame that the y coordinate is relative to. If None, the y coordinate is relative to the pose's frame.
        
        Returns:
            A new pose with the same x and z coordinates but a different y coordinate.
        """
        if frame is None:
            return self.with_position((self.x, y, self.z))
        else:
            in_frame = self.in_frame(frame)
            return in_frame.with_position((in_frame.x, y, in_frame.z)).in_frame(self.frame)
        
    def with_position_z(self, z: float, *, frame: Optional[str] = None) -> "Pose":
        """Return a new pose with the same x and y coordinates but a different z coordinate.
        
        Args:
            z: The new z coordinate of the pose.
            frame: The frame that the z coordinate is relative to. If None, the z coordinate is relative to the pose's frame.
        
        Returns:
            A new pose with the same x and y coordinates but a different z coordinate.
        """
        if frame is None:
            return self.with_position((self.x, self.y, z))
        else:
            in_frame = self.in_frame(frame)
            return in_frame.with_position((in_frame.x, in_frame.y, z)).in_frame(self.frame)
    
    def apply_transform(self, transform: "Transform") -> "Pose":
        """Return a new pose transformed by a transform.

        This method is provided to give compatibility with systems that talk about poses in terms of transforms,
        but it is usually more readable and debuggable to use a rotate/translate method instead.
        
        Args:
            transform: The transform to apply.
        
        Returns:
            A new pose transformed by a transform.
        """
        new_transform = self.transform * transform
        return Pose(new_transform, self.frame, self.pose_tree)
    
    
    def translate(self, translation: Sequence[float], *, frame: str = None) -> "Pose":
        """Return a new pose translated by a vector.

        The translation is relative to the basis of the pose being translated (i.e body-fixed) rather than the parent frame.
        That means that
        ```python
        p2 = p1.translate((1, 0, 0))
        ```
        and
        ```python
        p2 = p1.in_frame("some_other_frame").translate((1, 0, 0)).in_frame(p1.frame)
        ```
        are identical, because the parent frame of a pose does not affect its translation.

        Args:
            translation: The vector to translate by.
            frame: The frame that the vector is relative to.
        
        Returns:
            A new pose translated by a vector.
        """
        if frame is None:
            return Pose(Transform(self.position + self.rotation.apply(translation), self.rotation), self.frame, self.pose_tree)
        else:
            self_frame_t_frame = self.pose_tree.get_transform(self.frame, frame)
            return Pose(Transform(self.position + self_frame_t_frame.rotation.apply(translation), self.rotation), self.frame, self.pose_tree)
    
    def rotate_about_axis(self, axis: Sequence[float], angle: float, *, frame: Optional[str]=None) -> "Pose":
        """Return a new pose rotated about an axis.

        This is rotated in the basis of the pose being rotated (i.e body-fixed) rather than the parent frame, and it only
        rotates the orientation. This is *not* a rotation about the origin, but a change of the orientation of the pose.

        If you pass a frame, the axis will be interpreted as being expressed in that frame.
        
        Args:
            axis: The axis to rotate about.
            angle: The angle to rotate by, in radians.
            frame: The frame that the axis is relative to. If None, the axis is interpreted in the basis vectors of the pose (not the parent frame).
        
        Returns:
            A new pose with the same location but rotated about an axis.
        """
        axis = np.array(axis)
        if (np.inner(axis, axis) - 1) > 1e-6:
            raise ValueError("Axis must be a unit vector.")
        if frame is None:
            new_rotation = self.rotation * Rotation.from_rotvec(axis * angle)
        else:
            frame_t_self_frame = self.pose_tree.get_transform(frame, self.frame)
            new_rotation = Rotation.from_rotvec(frame_t_self_frame.rotation.apply(axis) * angle) * self.rotation
        return Pose(Transform(self.position, new_rotation), self.frame, self.pose_tree)
    
    def rotate_about_x(self, angle: float, *, frame: Optional[str]=None) -> "Pose":
        """Return a new pose rotated about the x axis.
        
        Args:
            angle: The angle to rotate by, in radians.
            frame: The frame that the axis is relative to. If not provided, it will rotate about the x-unit-vector of the pose being rotated.
        
        Returns:
            A new pose with the same location but rotated about the x axis.
        """
        return self.rotate_about_axis([1, 0, 0], angle, frame=frame)
    
    def rotate_about_y(self, angle: float, *, frame: Optional[str]=None) -> "Pose":
        """Return a new pose rotated about the y axis.
        
        Args:
            angle: The angle to rotate by, in radians.
            frame: The frame that the axis is relative to. If not provided, it will rotate about the y-unit-vector of the pose being rotated.
        
        Returns:
            A new pose with the same location but rotated about the y axis.
        """
        return self.rotate_about_axis([0, 1, 0], angle, frame=frame)
    
    def rotate_about_z(self, angle: float, *, frame: Optional[str]=None) -> "Pose":
        """Return a new pose rotated about the z axis.
        
        Args:
            angle: The angle to rotate by, in radians.
            frame: The frame that the axis is relative to. If not provided, it will rotate about the z-unit-vector of the pose being rotated.
        
        Returns:
            A new pose with the same location but rotated about the z axis.
        """
        return self.rotate_about_axis([0, 0, 1], angle, frame=frame)
    
    def point_x_at(self, target: "Pose", fixed_axis: str = None) -> "Pose":
        """Return a new pose rotated to point the x axis at a target pose.
        
        Args:
            target: The target pose to point at.
            fixed_axis: An optional axis ('x', 'y' or 'z') to keep fixed. If not provided,
                the pose will be rotated the minimum rotation to have the axis point at the target pose.
        
        Returns:
            A new pose with the same location but rotated to point the x axis at a target pose.
        """
        return self._point_at(target, [1, 0, 0], fixed_axis)
    
    def point_y_at(self, target: "Pose", fixed_axis: str = None) -> "Pose":
        """Return a new pose rotated to point the y axis at a target pose.
        
        Args:
            target: The target pose to point at.
            fixed_axis: An optional axis ('x', 'y' or 'z') to keep fixed. If not provided,
                the pose will be rotated the minimum rotation to have the axis point at the target pose.
        
        Returns:
            A new pose with the same location but rotated to point the y axis at a target pose.
        """
        return self._point_at(target, [0, 1, 0], fixed_axis)
    
    def point_z_at(self, target: "Pose", fixed_axis: str = None) -> "Pose":
        """Return a new pose rotated to point the z axis at a target pose.
        
        Args:
            target: The target pose to point at.
        
        Returns:
            A new pose with the same location but rotated to point the z axis at a target pose.
        """
        return self._point_at(target, [0, 0, 1], fixed_axis)
    
    def interpolate(self, target: "Pose", alpha: float) -> "Pose":
        """Return a new pose interpolated between two poses, in the frame of the first pose.
        
        Args:
            target: The target pose to interpolate to.
            alpha: The interpolation factor. 0.0 will return the current pose, 1.0 will return
                the target pose and values in between will return a pose interpolated between the two.
        
        Returns:
            A new pose interpolated between two poses.
        """
        return self.with_transform(self.transform.interpolate(target.in_frame(self.frame).transform, alpha))
    
    def distance_to(self, other: "Pose") -> float:
        """Return the cartesian distance between two poses.
        
        Args:
            other: The other pose.
        
        Returns:
            The distance between two poses.
        """
        return np.linalg.norm(self.position - other.in_frame(self.frame).position)
    
    def angle_to(self, target: "Pose") -> float:
        """Return the angle between two poses' orientations. Ignores position. Will convert to the same frame.
        
        This gives the magnitude of the minimal rotation that will align the two poses.

        Args:
            target: The target pose.
        
        Returns:
            The angle between two poses, in radians.
        """
        return self.transform.angle_to(target.in_frame(self.frame).transform)
    
    def _angle_about_axis_to(self, target: "Pose", axis: Sequence[float]) -> float:
        """Return the signed angle between two poses' orientations about an axis. Ignores position. Will convert to the same frame.
        
        Args:
            target: The target pose.
            axis: The axis to rotate about.
        
        Returns:
            The angle between two poses.

        Raises:
            ValueError: If the rotation is more than 20% out of the expected axis.
        """
        relative_rotation = self.transform.rotation.inv() * target.in_frame(self.frame).transform.rotation
        relative_rotvec = relative_rotation.as_rotvec()
        in_axis_rotation = np.dot(relative_rotvec, axis)
        out_of_axis_rotation = np.linalg.norm(relative_rotvec - in_axis_rotation * np.array(axis))
        if abs(in_axis_rotation) == 0 and abs(out_of_axis_rotation) > 0 or abs(out_of_axis_rotation) / abs(in_axis_rotation) > 0.2:
            raise ValueError("Rotation between poses is more than 20% out of expected axis.")
        return in_axis_rotation
    
    def angle_about_x_to(self, target: "Pose") -> float:
        """Return the signed angle between two poses' orientations about the x axis. Ignores position. Will convert to the same frame.
        
        Args:
            target: The target pose.
        
        Returns:
            The angle between two poses, in radians.

        Raises:
            ValueError: If the rotation is more than 20% out of the expected axis.
        """
        return self._angle_about_axis_to(target, [1, 0, 0]) 
    
    
    def angle_about_y_to(self, target: "Pose") -> float:
        """Return the signed angle between two poses' orientations about the y axis. Ignores position. Will convert to the same frame.
        
        Args:
            target: The target pose.
        
        Returns:
            The angle between two poses, in radians.

        Raises:
            ValueError: If the rotation is more than 20% out of the expected axis.
        """
        return self._angle_about_axis_to(target, [0, 1, 0])
    

    def angle_about_z_to(self, target: "Pose") -> float:
        """Return the signed angle between two poses' orientations about the z axis. Ignores position. Will convert to the same frame.
        
        Args:
            target: The target pose.
        
        Returns:
            The angle between two poses, in radians.

        Raises:
            ValueError: If the rotation is more than 20% out of the expected axis.
        """
        return self._angle_about_axis_to(target, [0, 0, 1])


    def _point_at(self, target: "Pose", axis: Sequence[float], fixed_axis: str=None)  -> "Pose":
        """Return a new pose rotated to point at a target pose.
        
        Args:
            target: The target pose to point at.
            axis: The axis to rotate about.
            fixed_axis: An optional axis ('x', 'y' or 'z') to keep fixed. If not provided,
                the pose will be rotated the minimum rotation to have the axis point at the target pose..
        
        Returns:
            A new pose with the same location but rotated to point at a target pose.
        
        """
        delta = target.in_frame(self.frame).position - self.position
        delta_local = self.rotation.inv().apply(delta)
        delta_rotation = Rotation.align_vectors(np.array([delta_local]), np.array([axis]))[0]
        if fixed_axis is None:
            return self.with_rotation(self.rotation * delta_rotation)
        if fixed_axis == 'x':
            ry, rz, rx = delta_rotation.as_euler('yzx')
            return self.with_rotation(self.rotation * Rotation.from_euler('xyz', [rx, 0, 0]))
        elif fixed_axis == 'y':
            rx, rz, ry = delta_rotation.as_euler('xzy')
            return self.with_rotation(self.rotation * Rotation.from_euler('yxz', [ry, 0, 0]))
        elif fixed_axis == 'z':
            rx, ry, rz = delta_rotation.as_euler('xyz')
            return self.with_rotation(self.rotation * Rotation.from_euler('zxy', [rz, 0, 0]))
        else:
            raise ValueError("fixed_axis must be one of 'x', 'y', or 'z'.")
    
    def __str__(self) -> str:
        return f"Pose(transform={self.transform}, frame={self.frame})"
    __repr__ = __str__


class PoseTree(ABC):
    """Abstract base class to manage poses and frames.
    
    To hook up this library to your project, you should subclass this class and hook it up to your stream of transforms
    by implementing _get_transform. Usually this will look like adding a callback-subscribe method to your class and storing
    the transforms between important frames as they come in. If you don't already have a way of interpolating between transforms,
    check out ROS's tf2 library. This library is designed to wrap nicely on top of tf2, if you are already using it.
    
    
    """
    def get_transform(self, parent_frame: str, child_frame: str, timestamp: float) -> "Transform":
        """Return the transform from the parent frame to the child frame."""
        return self._get_transform(parent_frame, child_frame, timestamp)
    
    def get_pose(self, position: Sequence[float], quaternion: Sequence[float], parent_frame: str) -> Pose:
        """Create a pose object from a position, quaternion, and parent frame. Essentially syntactic sugar on top of Pose's constructor."""
        return Pose(Transform.from_position_and_quaternion(position, quaternion), parent_frame, self)
    
    @contextmanager
    def temporary_frame(self, pose: Pose) -> str:
        """Create a temporary frame from a pose.
        
        This is intended to be used for anonymous frames for doing calculations.

        Example 1, find an approach angle that is close to the robot base:
        ```python
        for theta in np.linspace(0, 2 * np.pi, 100):
            with tree.temporary_frame(target_pose.rotate_about_z(theta)) as approach_frame:
                if abs(robot_base_pose.in_frame(approach_frame).y) < 0.1:
                    return theta
        ```

        Example 2, find a person that is ready to talk to the robot:
        ```python
        for person in people:
            with tree.temporary_frame(person.head_pose) as head_frame:
            x, y, _ = robot_base_pose.in_frame(head_frame).position
                if abs(y) < 0.5 and 0.5 < x < 1.5:
                    return person
        ```
        """
        name = uuid.uuid4().hex
        self.add_frame(pose, name)
        try:
            yield name
        finally:
            self.remove_frame(name)
    
    @abstractmethod
    def _get_transform(self, parent_frame: str, child_frame: str, timestamp: Optional[float] = None) -> "Transform":
        """Return the transform from the parent frame to the child frame. 
        
        Args:
            parent_frame: The name of the parent frame.
            child_frame: The name of the child frame.
            timestamp: The timestamp of the transform. If None, the latest transform is returned.
        """
        pass

    @abstractmethod
    def add_frame(self, pose: Pose, name: str) -> None:
        """Add a static frame to the tree.

        Example:
        ```python
        pose_tree.add_frame(table_corner_pose, 'table_frame')

        # Look at center of table:
        robot.look_at(
            Pose.from_position_and_rotation(
                [table_size_x / 2, tablesize_y_y / 2, 0],
                Rotation.identity(),
                'table_frame'))
        ```

        Don't use this for scratch frames for doing calculations. Use the temporary_frame context manager instead. This should 
        be used for permanant, human meaningful things like a workspace frame.

        This is also not a good solution for frames that are moving, as the frames will not be interpolated to particular timestamps.
        
        Args:
            pose: The pose of the new frame relative to the parent frame.
            name: The name of the new frame.
        """
        pass

    @abstractmethod
    def remove_frame(self, name: str) -> None:
        """Remove a static frame from the tree."""
        pass


class CustomFramePoseTree(PoseTree, ABC):
    """A abstract PoseTree implementation that implements custom frames.
    
    This is useful if your underlying pose structure doesn't support custom frames. You subclass this
    class and implement the _get_transform method and get full custom frame support. In a pinch you can implement
    _get_transform to raise a NotImplemented error and use 'add frame' but it is not particularly efficient and does
    not implement interpolation to particular timestamps, which will cause problems if you are getting sensor
    information off a moving robot.
    """
    def __init__(self) -> None:
        self.custom_frames = {}

    def add_frame(self, pose: Pose, name: str) -> None:
        """Add a frame to the tree. This is assumed to be a static frame, and will not be interpolated.

        If that frame is already present, this will update the frame to a new pose.

        Args:
            pose: The pose of the new frame relative to the parent frame.
            name: The name of the new frame.
        """
        self.custom_frames[name] = pose

    def remove_frame(self, name: str) -> None:
        """Remove a custom frame from the tree.
        
        This method is idempotent. If the frame is not present, it will do nothing.
        """
        if name in self.custom_frames:
            del self.custom_frames[name]

    def get_transform(self, parent_frame: str, child_frame: str, timestamp: Optional[float] = None) -> "Transform":
        """Return the transform from the parent frame to the child frame, at the timestamp."""
        if child_frame == parent_frame:
            return Transform.identity()
        if child_frame in self.custom_frames:
            child_root_pose_child = self.custom_frames[child_frame]
            parent_t_child_root = self.get_transform(parent_frame, child_root_pose_child.frame, timestamp)
            return parent_t_child_root * child_root_pose_child.transform
        elif parent_frame in self.custom_frames:
            parent_root_pose_parent = self.custom_frames[parent_frame]
            parent_root_t_child = self.get_transform(parent_root_pose_parent.frame, child_frame, timestamp)
            parent_t_parent_root = parent_root_pose_parent.transform.inverse
            return parent_t_parent_root * parent_root_t_child
        else:
            return self._get_transform(parent_frame, child_frame, timestamp)


    @abstractmethod
    def _get_transform(self, parent_frame: str, child_frame: str, timestamp: float) -> "Transform":
        """Return the transform from the parent frame to the child frame."""


class Transform(object):
    """A homogenous transform from a pose or frame to another location and rotation.
    
    In posetree, a Transform is a verb. It describes how to transform a pose from one frame to another.
    Almost all the time, you want to turn your raw data objects (coming from forward kinematics or perception or 
    localization) into a Pose object as soon as possible, and only go through Transforms at the borders of your sdk.

    Using Transform objects instead of Pose objects is usually a code smell, unless you're doing something really
    mathy with a bunch of intermediate transforms.
    """

    def __init__(self, position: Sequence[float], rotation: Rotation) -> None:
        """Create a transform from a position and rotation.
        
        Args:
            position: The position of the transform.
            rotation: The rotation of the transform.
        """
        self._position = np.array(position)
        self._rotation = rotation

    @classmethod
    def from_position_and_quaternion(cls, position: Sequence[float], quaternion: Sequence[float]) -> "Transform":
        """Create a transform from a position and quaternion.
        
        Args:
            position: The position of the transform.
            quaternion: The quaternion of the transform, in xyzw order.
        
        Returns:
            A new transform.
        """
        return cls(position, Rotation.from_quat(quaternion))

    @classmethod
    def identity(cls) -> "Transform":
        """Return the identity transform."""
        return Transform(np.zeros(3), Rotation.identity())

    @property
    def inverse(self) -> "Transform":
        """The inverse of the transform.
        
        How to get back from applying this transform.

        t1 * t1.inverse == Transform.identity()
        """
        return Transform(-self.rotation.inv().apply(self.position), self.rotation.inv())

    @property
    def position(self) -> np.ndarray:
        """The 1x3 cartesian position of the transform."""
        return self._position
    
    @property
    def rotation(self) -> Rotation:
        """The rotation component of the transform."""
        return self._rotation
    
    @property
    def x_axis(self) -> np.ndarray:
        """The x axis of the transform.
        
        This is the unit vector, in the parent frame, pointing in the new x direction, after applying the transform.
        """
        return self.rotation.apply(np.array([1, 0, 0]))
    
    @property
    def y_axis(self) -> np.ndarray:
        """The y axis of the transform.
        
        This is the unit vector, in the parent frame, pointing in the new y direction, after applying the transform.
        """
        return self.rotation.apply(np.array([0, 1, 0]))
    
    @property
    def z_axis(self) -> np.ndarray:
        """The z axis of the transform.
                
        This is the unit vector, in the parent frame, pointing in the new z direction, after applying the transform.
        """
        return self.rotation.apply(np.array([0, 0, 1])) 
    
    @property
    def x(self) -> float:
        """The x coordinate of the transform."""
        return self.position[0]
    
    @property
    def y(self) -> float:
        """The y coordinate of the transform."""
        return self.position[1]
    
    @property
    def z(self) -> float:
        """The z coordinate of the transform."""
        return self.position[2]
    
    def to_matrix(self) -> np.ndarray:
        """Return the 4x4 matrix representation of the transform."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.position
        return matrix
    
    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[float]]) -> "Transform":
        """Create a transform from a matrix.
        
        Args:
            matrix: The matrix of the transform.
        
        Returns:
            A new transform.
        """
        matrix = np.array(matrix)
        return Transform(matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]))

    def apply(self, other: Union["Transform", Sequence[float], Sequence[Sequence[float]]]) -> "Transform":
        """Multiply this transform by another transform or vector.

        If other is a transform, this is equivalent to applying this transform, then the other transform.
        If other is a vector, this is equivalent to applying this transform to the vector.
        
        Args:
            other: The transform or vector to multiply.
        
        Returns:
            The product of the two transforms.
        """
        if isinstance(other, Transform):
          return Transform(self.position + self.rotation.apply(other.position), self.rotation * other.rotation)
        else:
            other = np.array(other)
            return self._position + self.rotation.apply(other)
        
    __mul__ = apply
        
    def almost_equal(self, other: "Transform", atol: float=1e-8) -> bool:
        """Check if two transforms are almost equal.
        
        Handles floating point error in the position and rotation, and the fact that for quaternions negating 
        the vector gives the same rotation.
        """
        # Check if positions are almost equal.
        if not np.allclose(self._position, other._position, atol=atol):
            return False
        # Check if rotations are almost equal.
        # Since quaternion [x, y, z, w] represents the same rotation as [-x, -y, -z, -w], we also need to check the inverse quaternion.
        if not (np.allclose(self._rotation.as_quat(), other._rotation.as_quat(), atol=atol)
                or np.allclose(self._rotation.as_quat(), -other._rotation.as_quat(), atol=atol)):
            return False
        return True
    
    def interpolate(self, other: "Transform", alpha: float) -> "Transform":
        """Interpolate between two transforms.
        
        Args:
            other: The other transform to interpolate with.
            alpha: The interpolation factor. 0 gives this transform, 1 gives the other transform.
        
        Returns:
            The interpolated transform.
        """
        # Combine the rotations into a single Rotation object
        rotations = Rotation.from_quat(np.vstack((self.rotation.as_quat(), other.rotation.as_quat())))
        slerp = Slerp([0, 1], rotations)    
        new_rotation= slerp([alpha])[0]
        return Transform(self.position * (1 - alpha) + other.position * alpha, new_rotation)
    
    def angle_to(self, target: "Transform") -> float:
        """Return the angle between two transforms' orientations. Ignores position.
        
        Args:
            target: The target transform.
        
        Returns:
            The angle between two transform, in radians.
        """
        relative_rotation = self.rotation.inv() * target.rotation
        return np.linalg.norm(relative_rotation.as_rotvec())
    
    def __eq__(self, other: "Transform") -> bool:
        """Check if two transforms are equal."""
        return self.almost_equal(other, atol=0)

    
    def __str__(self) -> str:
        return f"Transform(position={self.position}, rotation={self.rotation.as_quat()})"
    __repr__ = __str__
            

