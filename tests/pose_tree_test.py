import numpy as np
import pytest
from pose import Transform, CustomFramePoseTree, Pose
from scipy.spatial.transform import Rotation

class TestPoseTree(CustomFramePoseTree):
    def _get_transform(self, parent_frame: str, child_frame: str, timestamp: float) -> Transform:
        raise NotImplementedError(f"Non-custom frames not implemented in TestPoseTree: '{parent_frame}' -> '{child_frame}'")

def test_pose_tree():
    pose_tree = TestPoseTree()
    identity_transform = Transform.identity()

    # Add some frames
    pose_tree.add_frame(Pose(identity_transform, "parent", pose_tree), "child1")
    pose_tree.add_frame(Pose(identity_transform, "child1", pose_tree), "child2")

    # Test getting transforms
    assert pose_tree.get_transform("parent", "parent", 0) == identity_transform
    assert pose_tree.get_transform("child1", "child1", 0) == identity_transform
    assert pose_tree.get_transform("child2", "child2", 0) == identity_transform
    assert pose_tree.get_transform("parent", "child1", 0) == identity_transform
    assert pose_tree.get_transform("child1", "child2", 0) == identity_transform
    assert pose_tree.get_transform("parent", "child2", 0) == identity_transform

    # Test removing a frame
    pose_tree.remove_frame("child1")
    with pytest.raises(NotImplementedError):
        pose_tree.get_transform("parent", "child1", 0)

def test_single_hop():
    pose_tree = TestPoseTree()

    # Define a known transform
    rotation = Rotation.from_euler('xyz', [0.1, 0.2, 0.3], degrees=True)
    position = np.array([1, 2, 3])
    known_transform = Transform(position, rotation)

    # Add a child to the root with the known transform
    pose_tree.add_frame(Pose(known_transform, "root", pose_tree), "child")

    # The transform from the root to the child should be the known transform
    assert Transform.almost_equal(pose_tree.get_transform("root", "child", 0), known_transform)

    # The transform from the child to the root should be the inverse of the known transform
    assert Transform.almost_equal(pose_tree.get_transform("child", "root", 0), known_transform.inverse)

def test_double_hop():
    pose_tree = TestPoseTree()

    # Define two known transforms
    rotation1 = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
    position1 = np.array([1, 0, 0])
    known_transform1 = Transform(position1, rotation1)

    rotation2 = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    position2 = np.array([10, 0, 0])
    known_transform2 = Transform(position2, rotation2)

    # Add a child to the root with the first known transform
    pose_tree.add_frame(Pose(known_transform1, "root", pose_tree), "child")

    # Add a grandchild to the child with the second known transform
    pose_tree.add_frame(Pose(known_transform2, "child", pose_tree), "grandchild")

    # The transform from the root to the grandchild should be the result of combining the two known transforms
    expected_transform = known_transform1 * known_transform2
    assert Transform.almost_equal(pose_tree.get_transform("root", "grandchild", 0), expected_transform)

    # The transform from the grandchild to the root should be the inverse of the expected transform
    assert Transform.almost_equal(pose_tree.get_transform("grandchild", "root", 0), expected_transform.inverse)


def test_siblings():
    pose_tree = TestPoseTree()

    # Define two known transforms
    rotation1 = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
    position1 = np.array([1, 0, 0])
    known_transform1 = Transform(position1, rotation1)

    rotation2 = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
    position2 = np.array([10, 0, 0])
    known_transform2 = Transform(position2, rotation2)

    # Add a children to the root with known transforms
    pose_tree.add_frame(Pose(known_transform1, "root", pose_tree), "child1")
    pose_tree.add_frame(Pose(known_transform2, "root", pose_tree), "child2")

    # The difference should be 9 in the y direction
    expected_transform =Transform([0, -9, 0], Rotation.identity())
    assert Transform.almost_equal(pose_tree.get_transform("child1", "child2", 0), expected_transform)
    assert Transform.almost_equal(pose_tree.get_transform("child2", "child1", 0), expected_transform.inverse)


def test_pose_tree_complex_transforms():
    pose_tree = TestPoseTree()
    
    # Define some non-identity transforms
    parent_t_child1 = Transform.from_position_and_quaternion([1, 2, 3], [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    child1_t_child2 = Transform.from_position_and_quaternion([4, 5, 6], [0, np.sin(np.pi/6), 0, np.cos(np.pi/6)])
    parent_t_child3 = Transform.from_position_and_quaternion([7, 8, 9], [np.sin(np.pi/8), 0, 0, np.cos(np.pi/8)])
    child3_t_child4 = Transform.from_position_and_quaternion([10, 11, 12], [np.sin(np.pi/10), 0, np.sin(np.pi/10), np.cos(np.pi/10)])

    # Add some frames
    pose_tree.add_frame(Pose(parent_t_child1, "parent", pose_tree), "child1")
    pose_tree.add_frame(Pose(child1_t_child2, "child1", pose_tree), "child2")
    pose_tree.add_frame(Pose(parent_t_child3, "parent", pose_tree), "child3")
    pose_tree.add_frame(Pose(child3_t_child4, "child3", pose_tree), "child4")

    # Calculate expected transformations
    child2_t_child4 = child1_t_child2.inverse * parent_t_child1.inverse * parent_t_child3 * child3_t_child4

    assert Transform.almost_equal(pose_tree.get_transform("child2", "child4", 0), child2_t_child4)
    assert Transform.almost_equal(pose_tree.get_transform("child4", "child2", 0), child2_t_child4.inverse)


def test_temporary_frame():
    pose_tree = TestPoseTree()
    # Define two known transforms
    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    position = np.array([0, 0, 0])
    known_transform = Transform(position, rotation)
    pose_tree.add_frame(Pose(known_transform, "root", pose_tree), "child")

    p = Pose(known_transform, "root", pose_tree)

    num_frames = len(pose_tree.custom_frames)

    for i in range(10):
        p = Pose(known_transform, "child", pose_tree)
        with pose_tree.temporary_frame(p.translate([i, 0, 0])) as work_frame:
            assert p.in_frame(work_frame).x == -i

    assert len(pose_tree.custom_frames) == num_frames
