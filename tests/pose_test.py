import numpy as np
import pytest
from typing import Optional
from pytest import approx
from posetree import Transform, CustomFramePoseTree, Pose
from scipy.spatial.transform import Rotation

class TestPoseTree(CustomFramePoseTree):
    def _get_transform(self, parent_frame: str, child_frame: str, timestamp: Optional[float] = None) -> Transform:
        raise NotImplementedError("Non-custom frames not implemented in TestPoseTree")

def get_pose_tree() -> TestPoseTree:
    pose_tree = TestPoseTree()

    # Robot frame is at [2, 1, 0] in world frame.
    robot_position = np.array([2, 1, 0])
    robot_rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    robot_transform = Transform(robot_position, robot_rotation)
    pose_tree.add_frame(Pose(robot_transform, "world", pose_tree), "robot")

    # Camera frame is 1 meter up in z in robot frame and has z facing forward (towards robot x).
    camera_position = np.array([0, 0, 1])
    camera_rotation = Rotation.from_euler('yxz', [90, 0, 0], degrees=True)
    camera_transform = Transform(camera_position, camera_rotation)
    pose_tree.add_frame(Pose(camera_transform, "robot", pose_tree), "camera")

    # Tool frame is at [0.5, 0.5, 0.75] in robot with z facing down.
    tool_position = np.array([0.5, 0.5, 0.75])
    tool_rotation = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
    tool_transform = Transform(tool_position, tool_rotation)
    pose_tree.add_frame(Pose(tool_transform, "robot", pose_tree), "tool")

    return pose_tree

def assert_rotations_equal(rotation1: Rotation, rotation2: Rotation):
    assert np.allclose(rotation1.as_quat(), rotation2.as_quat()) or np.allclose(rotation1.as_quat(), -rotation2.as_quat())

def test_pose():
    pose_tree = get_pose_tree()
    transform = Transform(np.array([1, 2, 3]), Rotation.from_quat([0, 0, 0, 1]))
    pose = Pose(transform, 'robot', pose_tree)
    
    assert np.array_equal(pose.position, np.array([1, 2, 3]))
    assert_rotations_equal(pose.rotation, Rotation.from_quat([0, 0, 0, 1]))
    assert pose.transform == transform
    assert pose.frame == 'robot'
    assert pose.pose_tree == pose_tree


def test_get_pose():
    pose_tree = get_pose_tree()

    # Define a position, quaternion, and parent frame
    position = np.array([3, 2, 1])
    quaternion = [0, 0, 0, 1]  # Identity quaternion (no rotation)
    parent_frame = 'world'

    # Use get_pose to create the pose
    pose = pose_tree.get_pose(position, quaternion, parent_frame)

    # Check the pose's properties
    assert np.array_equal(pose.position, position)
    assert_rotations_equal(pose.rotation, Rotation.from_quat(quaternion))
    assert pose.frame == parent_frame

    # Define a non-identity quaternion
    quaternion = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]  # 90 degree rotation around z axis

    # Create another pose with the non-identity quaternion
    pose = pose_tree.get_pose(position, quaternion, parent_frame)

    # Check the new pose's properties
    assert np.array_equal(pose.position, position)
    assert_rotations_equal(pose.rotation, Rotation.from_quat(quaternion))
    assert pose.frame == parent_frame


def test_position_props():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    assert pose.x == 1
    assert pose.y == 2
    assert pose.z == 3

def test_in_frame():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 3], [0, 0, 0, 1], 'camera')
    pose_in_robot = pose.in_frame('robot')
    assert pose_in_robot.frame == 'robot'
    assert pose_in_robot.x == approx(3)
    assert pose_in_robot.y == approx(0)
    assert pose_in_robot.z == approx(1)

    # The 'z' in camera frame is the 'x' in robot frame
    assert pose_in_robot.z_axis == approx([1, 0, 0])
    assert pose_in_robot.y_axis == approx([0, 1, 0])

def test_in_frame_same_frame():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 3], [0, 0, 0, 1], 'camera')
    pose_in_camera = pose.in_frame('camera')
    assert pose_in_camera.frame == 'camera'
    assert pose.transform == pose_in_camera.transform

def test_with_position():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 3], [0, 0, 0, 1], 'camera')
    new_pose = pose.with_position([1, 2, 4])

    # assert new pose correct
    assert new_pose.position == pytest.approx(np.array([1, 2, 4]))
    assert_rotations_equal(new_pose.rotation, Rotation.from_quat([0, 0, 0, 1]))

    # assert old pose unchanged
    assert pose.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(pose.rotation, Rotation.from_quat([0, 0, 0, 1]))

def test_with_position_x():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([2, 3, 4], [0, 0, 0, 1], 'robot')
    new_pose = pose.with_position_x(10)

    # assert new pose's x is updated correctly
    assert new_pose.x == 10
    # assert other components remain the same
    assert new_pose.y == pose.y
    assert new_pose.z == pose.z
    assert_rotations_equal(new_pose.rotation, pose.rotation)

def test_with_position_y():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([2, 3, 4], [0, 0, 0, 1], 'robot')
    new_pose = pose.with_position_y(10)

    # assert new pose's y is updated correctly
    assert new_pose.y == 10
    # assert other components remain the same
    assert new_pose.x == pose.x
    assert new_pose.z == pose.z
    assert_rotations_equal(new_pose.rotation, pose.rotation)

def test_with_position_z():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([2, 3, 4], [0, 0, 0, 1], 'robot')
    new_pose = pose.with_position_z(10)

    # assert new pose's z is updated correctly
    assert new_pose.z == 10
    # assert other components remain the same
    assert new_pose.x == pose.x
    assert new_pose.y == pose.y
    assert_rotations_equal(new_pose.rotation, pose.rotation)

def test_with_position_xyz_in_frame():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([2, 3, 4], [0, 0, 0, 1], 'robot')
    
    z1_in_camera = pose.with_position_z(1, frame='camera')
    assert z1_in_camera.x == pytest.approx(1)
    assert z1_in_camera.y == pytest.approx(3)
    assert z1_in_camera.z == pytest.approx(4)

def test_with_rotation():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 3], [0, 0, 0, 1], 'camera')
    new_rotation = Rotation.from_euler('xyz', [45, 45, 45], degrees=True)
    new_pose = pose.with_rotation(new_rotation)
    
    # assert new pose correct
    assert new_pose.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(new_pose.rotation, new_rotation)

    # assert old pose unchanged
    assert pose.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(pose.rotation, Rotation.from_quat([0, 0, 0, 1]))


def test_with_rotation_matching():
    pose_tree = get_pose_tree()
    
    pose1 = pose_tree.get_pose([0, 0, 3], [0, 0, 0, 1], 'camera')
    pose2 = pose_tree.get_pose([1, 2, 4], [0, 0, 1, 0], 'camera')
    
    new_pose = pose1.with_rotation_matching(pose2)

    # assert new pose correct
    assert new_pose.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(new_pose.rotation, Rotation.from_quat([0, 0, 1, 0]))

    # assert old poses unchanged
    assert pose1.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(pose1.rotation, Rotation.from_quat([0, 0, 0, 1]))

    assert pose2.position == pytest.approx(np.array([1, 2, 4]))
    assert_rotations_equal(pose2.rotation, Rotation.from_quat([0, 0, 1, 0]))

def test_with_rotation_matching_in_other_frame():
    pose_tree = get_pose_tree()
    
    pose1 = pose_tree.get_pose([0, 0, 3], [0, 0, 0, 1], 'camera')
    pose2 = pose_tree.get_pose([1, 2, 4], [0, 0, 1, 0], 'camera').in_frame('robot')
    
    new_pose = pose1.with_rotation_matching(pose2)

    # assert new pose correct
    assert new_pose.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(new_pose.rotation, Rotation.from_quat([0, 0, 1, 0]))

    # assert old poses unchanged
    assert pose1.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(pose1.rotation, Rotation.from_quat([0, 0, 0, 1]))

    # Pose 2 is different because it is in a different frame
    assert pose2.position != pytest.approx(np.array([1, 2, 4]))
    assert pose2.in_frame("camera").position == pytest.approx(np.array([1, 2, 4]))
    assert_rotations_equal(pose2.in_frame("camera").rotation, Rotation.from_quat([0, 0, 1, 0]))

def test_apply_transform():
    pose_tree = get_pose_tree()
    
    pose = pose_tree.get_pose([0, 0, 3], [0, 0, 0, 1], 'camera')
    transform = Transform(np.array([1, 2, -1]), Rotation.from_quat([0, 1, 0, 0]))
    
    new_pose = pose.apply_transform(transform)

    # assert new pose correct
    assert new_pose.position == pytest.approx(np.array([1, 2, 2]))  # old position + transformation position
    assert_rotations_equal(new_pose.rotation, Rotation.from_quat([0, 1, 0, 0]))  # transformation rotation

    # assert old pose unchanged
    assert pose.position == pytest.approx(np.array([0, 0, 3]))
    assert_rotations_equal(pose.rotation, Rotation.from_quat([0, 0, 0, 1]))

def test_translate():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')

    # Apply a translation
    new_pose = pose.translate([2, -1, 0])

    # Assert new pose correct
    assert new_pose.position == pytest.approx(np.array([3, 1, 3]))
    assert_rotations_equal(new_pose.rotation, Rotation.from_quat([0, 0, 0, 1]))

    # Assert old pose unchanged
    assert pose.position == pytest.approx(np.array([1, 2, 3]))
    assert_rotations_equal(pose.rotation, Rotation.from_quat([0, 0, 0, 1]))
    assert pose.frame == 'robot'

def test_translate_non_identity():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot').rotate_about_z(np.pi/2)

    # Apply a translation
    new_pose = pose.translate([1, 0, 10])

    # Assert new pose correct
    assert new_pose.position == pytest.approx(np.array([1, 3, 13]))
    assert_rotations_equal(new_pose.rotation, pose.rotation)

def test_translate_in_frame():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 2], [0, 0, 0, 1], 'camera')

    # Apply a translation
    new_pose = pose.translate([0, 0, 1], frame='robot')

    # Assert new pose correct
    assert new_pose.position == pytest.approx(np.array([-1, 0, 2]))
    assert_rotations_equal(new_pose.rotation, Rotation.from_quat([0, 0, 0, 1]))


def test_rotate_about_axis():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    
    # Rotate 90 degrees about z-axis
    new_pose = pose.rotate_about_axis([0, 0, 1], np.pi/2)
    
    # Check position - should remain the same
    assert new_pose.position == pytest.approx(np.array([1, 2, 3]))
    
    # Check rotation - should be 90 degrees about z-axis
    expected_rotation = Rotation.from_euler('z', np.pi/2)
    assert_rotations_equal(new_pose.rotation, expected_rotation)


def test_rotate_about_axis_chaining():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    
    # Rotate 90 degrees about y-axis then 90 degrees about z-axis
    new_pose = pose.rotate_about_axis([0, 1, 0], np.pi/2).rotate_about_axis([0, 0, 1], np.pi/2)
    
    # Check position - should remain the same
    assert new_pose.position == pytest.approx(np.array([1, 2, 3]))
    
    assert new_pose.x_axis == pytest.approx(np.array([0, 1, 0]))
    assert new_pose.y_axis == pytest.approx(np.array([0, 0, 1]))
    assert new_pose.z_axis == pytest.approx(np.array([1, 0, 0]))

def test_rotate_about_in_frame():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    
    # Rotate 90 degrees about y-axis then 90 degrees about z-axis
    new_pose = pose.rotate_about_axis([0, 0, 1], np.pi/2, frame="camera")

    assert new_pose.x_axis == pytest.approx(np.array([1, 0, 0]))
    assert new_pose.y_axis == pytest.approx(np.array([0, 0, -1]))

def test_rotate_about_in_frame_non_identity_start():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot').rotate_about_axis([0, 0, 1], np.pi/2)
    
    # Rotate 90 degrees about y-axis then 90 degrees about z-axis
    new_pose = pose.rotate_about_axis([0, 0, 1], np.pi/2, frame="camera")

    assert new_pose.x_axis == pytest.approx(np.array([0, 0, -1]))
    assert new_pose.y_axis == pytest.approx(np.array([-1, 0, 0]))

def test_rotate_about_x():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    
    # Rotate 90 degrees about x-axis
    new_pose = pose.rotate_about_x(np.pi/2)
    
    # Check position - should remain the same
    assert new_pose.position == pytest.approx(np.array([1, 2, 3]))
    
    # Check rotation - should be 90 degrees about x-axis
    expected_rotation = Rotation.from_euler('x', np.pi/2)
    assert_rotations_equal(new_pose.rotation, expected_rotation)

def test_rotate_about_y():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    
    # Rotate 90 degrees about y-axis
    new_pose = pose.rotate_about_y(np.pi/2)
    
    # Check position - should remain the same
    assert new_pose.position == pytest.approx(np.array([1, 2, 3]))
    
    # Check rotation - should be 90 degrees about y-axis
    expected_rotation = Rotation.from_euler('y', np.pi/2)
    assert_rotations_equal(new_pose.rotation, expected_rotation)

def test_rotate_about_z():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    
    # Rotate 90 degrees about z-axis
    new_pose = pose.rotate_about_z(np.pi/2)
    
    # Check position - should remain the same
    assert new_pose.position == pytest.approx(np.array([1, 2, 3]))
    
    # Check rotation - should be 90 degrees about z-axis
    expected_rotation = Rotation.from_euler('z', np.pi/2)
    assert_rotations_equal(new_pose.rotation, expected_rotation)

def test_rotate_about_axis_degrees():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')
    
    # Rotate 90 degrees about z-axis
    new_pose_z = pose.rotate_about_z(90, degrees=True)
    new_pose_y = pose.rotate_about_y(90, degrees=True)
    new_pose_x = pose.rotate_about_x(90, degrees=True)
    
    # Check position - should remain the same
    assert new_pose_z.position == pytest.approx(np.array([1, 2, 3]))
    assert new_pose_y.position == pytest.approx(np.array([1, 2, 3]))
    assert new_pose_x.position == pytest.approx(np.array([1, 2, 3]))
    
    # Check rotation - should be 90 degrees about z-axis
    for p, axis in zip([new_pose_z, new_pose_y, new_pose_x], ['z', 'y', 'x']):
        expected_rotation = Rotation.from_euler(axis, np.pi/2)
        assert_rotations_equal(p.rotation, expected_rotation)

def test_rotate_about_axis_raises_on_non_unit_axis():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([1, 2, 3], [0, 0, 0, 1], 'robot')

    with pytest.raises(ValueError):
        pose.rotate_about_axis([0, 0, 2], np.pi/2)

def test_distance_to():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot')
    
    distance = pose.distance_to(pose2)
    assert distance == pytest.approx(4)

def test_distance_to_different_frame():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').in_frame('camera')
    
    distance = pose.distance_to(pose2)
    assert distance == pytest.approx(4)

def test_angle_to():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').rotate_about_x(np.pi/2)
    
    angle = pose.angle_to(pose2)
    assert angle == pytest.approx(np.pi/2)


def test_angle_to2():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_x(-np.pi/2)
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').rotate_about_x(np.pi/2)
    
    angle = pose.angle_to(pose2)
    assert angle == pytest.approx(np.pi)

def test_angle_to3():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_x(-np.pi/2)
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').rotate_about_y(.1)
    
    angle = pose.angle_to(pose2)
    assert angle > np.pi/2

def test_angle_about_z_to():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').rotate_about_z(-np.pi/2)
    
    angle = pose.angle_about_z_to(pose2)
    assert angle == pytest.approx(-np.pi/2)

def test_angle_about_z_to_other_frames():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').rotate_about_z(-np.pi/2).in_frame('camera')
    
    angle = pose.angle_about_z_to(pose2)
    assert angle == pytest.approx(-np.pi/2)

def test_angle_about_z_to_slight_misalignment():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').rotate_about_x(0.001).rotate_about_z(-np.pi/2)
    
    angle = pose.angle_about_z_to(pose2)
    assert angle == pytest.approx(-np.pi/2)

def test_angle_about_z_to_no_difference():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_z(-np.pi/2)
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').rotate_about_z(-np.pi/2)
    
    angle = pose.angle_about_z_to(pose2)
    assert angle == pytest.approx(0)

def test_angle_about_z_180():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_z(np.pi)
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot')

    angle = pose.angle_about_z_to(pose2)
    assert angle == pytest.approx(np.pi) or angle == pytest.approx(-np.pi)

def test_angle_about_z_but_zs_90_apart():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_x(np.pi/2)
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot')
    
    # assert Raises Value Error
    with pytest.raises(ValueError):
        pose.angle_about_z_to(pose2)

def test_point_at():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot')
    
    pointed_x = pose.point_x_at(pose2)
    assert pointed_x.x_axis == pytest.approx(np.array([0, 1, 0]))

    pointed_y = pose.point_y_at(pose2)
    assert pointed_y.y_axis == pytest.approx(np.array([0, 1, 0]))

    pointed_z = pose.point_z_at(pose2)
    assert pointed_z.z_axis == pytest.approx(np.array([0, 1, 0]))

def test_point_at_non_identity_start():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_z(3)
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot')
    
    pointed_x = pose.point_x_at(pose2)
    assert pointed_x.x_axis == pytest.approx(np.array([0, 1, 0]))

    pointed_y = pose.point_y_at(pose2)
    assert pointed_y.y_axis == pytest.approx(np.array([0, 1, 0]))

    pointed_z = pose.point_z_at(pose2)
    assert pointed_z.z_axis == pytest.approx(np.array([0, 1, 0]))

def test_point_at_different_frame():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 0], [0, 0, 0, 1], 'robot').in_frame('camera')
    
    pointed_x = pose.point_x_at(pose2)
    assert pointed_x.x_axis == pytest.approx(np.array([0, 1, 0]))

    pointed_y = pose.point_y_at(pose2)
    assert pointed_y.y_axis == pytest.approx(np.array([0, 1, 0]))

    pointed_z = pose.point_z_at(pose2)
    assert pointed_z.z_axis == pytest.approx(np.array([0, 1, 0]))

def test_point_at_self():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    
    pointed_x = pose.point_x_at(pose)
    assert pointed_x.x_axis == pytest.approx(np.array([1, 0, 0]))

    pointed_y = pose.point_y_at(pose)
    assert pointed_y.y_axis == pytest.approx(np.array([0, 1, 0]))

    pointed_z = pose.point_z_at(pose)
    assert pointed_z.z_axis == pytest.approx(np.array([0, 0, 1]))

def test_point_at_180_degrees():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([-5, 0, 0], [0, 0, 0, 1], 'robot')
    
    pointed_x = pose.point_x_at(pose2)
    assert pointed_x.x_axis == pytest.approx(np.array([-1, 0, 0]))

def test_point_x_at_with_fixed_z_axis():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 3], [0, 0, 0, 1], 'robot')
    
    pointed_x = pose.point_x_at(pose2, fixed_axis='z')
    assert pointed_x.x_axis == pytest.approx(np.array([0, 1, 0]))
    assert pointed_x.z_axis == pytest.approx(np.array([0, 0, 1]))

def test_point_x_at_with_fixed_z_axis_non_identity_start():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_z(3)
    pose2 = pose_tree.get_pose([0, 4, 3], [0, 0, 0, 1], 'robot')
    
    pointed_x = pose.point_x_at(pose2, fixed_axis='z')
    assert pointed_x.x_axis == pytest.approx(np.array([0, 1, 0]))
    assert pointed_x.z_axis == pytest.approx(np.array([0, 0, 1]))

def test_point_x_at_with_fixed_y_axis():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([0, 4, 3], [0, 0, 0, 1], 'robot')
    
    pointed_x = pose.point_x_at(pose2, fixed_axis='y')
    assert pointed_x.x_axis == pytest.approx(np.array([0, 0, 1]))
    assert pointed_x.y_axis == pytest.approx(np.array([0, 1, 0]))

def test_point_z_at_with_fixed_x_axis():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot')
    pose2 = pose_tree.get_pose([2, 4, 0], [0, 0, 0, 1], 'robot')
    
    pointed_z = pose.point_z_at(pose2, fixed_axis='x')
    assert pointed_z.z_axis == pytest.approx(np.array([0, 1, 0]))
    assert pointed_z.x_axis == pytest.approx(np.array([1, 0, 0]))

def test_point_at_with_fixed_axis_non_aligned():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_y(0.2)
    pose2 = pose_tree.get_pose([0, 4, 5], [0, 0, 0, 1], 'robot')

    pointed_x = pose.point_x_at(pose2, fixed_axis='z')
    assert pointed_x.z_axis == pytest.approx(pose.z_axis)

    # Check that small rotations result in less pointing

    initial_alignment = pointed_x.x_axis.dot([0, 1, 0])
    assert initial_alignment < 0.99

    for delta in [-0.01, 0.01]:
        for axis in [[1,0,0], [0,1,0], [0,0,1]]:
            test_pose = pose.rotate_about_axis(axis, delta)
            new_alignment = test_pose.x_axis.dot([0, 1, 0])
            assert new_alignment < initial_alignment

def test_point_at_with_fixed_axis_non_aligned_non_identity_start():
    pose_tree = get_pose_tree()
    pose = pose_tree.get_pose([0, 0, 0], [0, 0, 0, 1], 'robot').rotate_about_y(0.2).rotate_about_z(0.3).rotate_about_x(0.4)
    pose2 = pose_tree.get_pose([0, 4, 5], [0, 0, 0, 1], 'robot')

    pointed_x = pose.point_x_at(pose2, fixed_axis='z')
    assert pointed_x.z_axis == pytest.approx(pose.z_axis)

    initial_alignment = pointed_x.x_axis.dot([0, 1, 0])
    assert initial_alignment < 0.99

    for delta in [-0.01, 0.01]:
        for axis in [[1,0,0], [0,1,0], [0,0,1]]:
            test_pose = pose.rotate_about_axis(axis, delta)
            new_alignment = test_pose.x_axis.dot([0, 1, 0])
            assert new_alignment < initial_alignment

    