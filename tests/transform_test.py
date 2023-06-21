import numpy as np
import pytest
from pytest import approx
from posetree import Transform
from scipy.spatial.transform import Rotation


def test_transform_init():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)

    assert np.all(transform._position == np.array(position))
    assert transform._rotation == rotation


def test_transform_identity():
    identity = Transform.identity()

    assert np.all(identity._position == np.zeros(3))
    assert np.allclose(identity._rotation.as_quat(), Rotation.identity().as_quat())



def test_transform_inverse():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)
    inverse = transform.inverse

    assert np.all(inverse._position == approx(-rotation.inv().apply(position)))
    assert np.allclose(inverse._rotation.as_quat(), rotation.inv().as_quat())

def test_transform_inverse_multiplication():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)

    product = transform * transform.inverse

    assert product.almost_equal(Transform.identity())

def test_transform_position():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)

    assert np.all(transform.position == np.array(position))

def test_transform_rotation():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)

    assert transform.rotation == rotation

def test_transform_axis():
    position = [0, 0, 0]
    rotation = Rotation.from_euler('xyz', [90, 0, 0], degrees=True)
    transform = Transform(position, rotation)

    assert np.allclose(transform.x_axis, np.array([1, 0, 0]))
    assert np.allclose(transform.y_axis, np.array([0, 0, 1]))
    assert np.allclose(transform.z_axis, np.array([0, -1, 0]))

def test_transform_coordinates():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)

    assert transform.x == position[0]
    assert transform.y == position[1]
    assert transform.z == position[2]


def test_transform_from_position_and_quaternion():
    position = [1, 2, 3]
    quaternion = [0, 0, 0, 1]
    transform = Transform.from_position_and_quaternion(position, quaternion)

    expected_transform = Transform(position, Rotation.from_quat(quaternion))

    assert Transform.almost_equal(transform, expected_transform)



def test_transform_to_matrix():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position

    assert np.all(transform.to_matrix() == matrix)


def test_transform_from_matrix():
    position = [1, 2, 3]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(position, rotation)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position

    new_transform = Transform.from_matrix(matrix)

    assert Transform.almost_equal(new_transform, transform)

def test_transform_mul_transform():
    position1 = [1, 2, 3]
    quaternion1 = [0, 0, 0, 1]
    transform1 = Transform.from_position_and_quaternion(position1, quaternion1)

    position2 = [4, 5, 6]
    quaternion2 = [0, 1, 0, 0]
    transform2 = Transform.from_position_and_quaternion(position2, quaternion2)

    result = transform1 * transform2

    expected_position = np.add(position1, Rotation.from_quat(quaternion1).apply(position2))
    expected_rotation = Rotation.from_quat(quaternion1) * Rotation.from_quat(quaternion2)

    expected_transform = Transform(expected_position, expected_rotation)

    assert Transform.almost_equal(result, expected_transform)


def test_transform_mul_sequence():
    position = [1, 2, 3]
    quaternion = [0, 0, 0, 1]
    transform = Transform.from_position_and_quaternion(position, quaternion)

    sequence = [4, 5, 6]

    result = transform * sequence

    expected_result = np.add(position, Rotation.from_quat(quaternion).apply(sequence))

    assert np.allclose(result, expected_result)


def test_transform_mul_sequence_2():
    position = [0, 0, 0]
    transform = Transform(position=position, rotation=Rotation.from_euler('xyz', [0, 0, 90], degrees=True))
    sequence = [1, 0, 0]

    result = transform * sequence

    expected_result = [0, 1, 0]

    assert np.allclose(result, expected_result)


def test_interpolate():
    # Create two transform objects
    t1 = Transform(position=np.array([0, 0, 0]), rotation=Rotation.from_quat([0, 0, 0, 1]))
    t2 = Transform(position=np.array([1, 1, 1]), rotation=Rotation.from_quat([0, 1, 0, 0]))

    # Interpolate between the transforms at alpha=0.0 (should equal t1)
    t_interpolated = t1.interpolate(t2, 0.0)
    assert t_interpolated.almost_equal(t1)

    # Interpolate between the transforms at alpha=1.0 (should equal t2)
    t_interpolated = t1.interpolate(t2, 1.0)
    assert t_interpolated.almost_equal(t2)

    # Interpolate between the transforms at alpha=0.5 (should be midway between t1 and t2)
    t_interpolated = t1.interpolate(t2, 0.5)
    expected_position = np.array([0.5, 0.5, 0.5])  # Midway between t1.position and t2.position
    assert np.allclose(t_interpolated.position, expected_position)
    assert t1.angle_to(t_interpolated) == approx(t2.angle_to(t_interpolated))  # Interpolated rotation should be halfway between t1.rotation and t2.rotation


def test_zero_position_transform():
    zero_position = [0, 0, 0]
    rotation = Rotation.from_quat([0, 0, 0, 1])
    transform = Transform(zero_position, rotation)
    assert Transform.almost_equal(transform.inverse, Transform.identity())

def test_identity_rotation_transform():
    position = [1, 2, 3]
    identity_rotation = Rotation.identity()
    transform = Transform(position, identity_rotation)
    identity_transform = Transform(-np.array(position), identity_rotation)
    assert Transform.almost_equal(transform.inverse, identity_transform)

def test_transform_mul_invalid_sequence():
    position = [1, 2, 3]
    quaternion = [0, 0, 0, 1]
    transform = Transform.from_position_and_quaternion(position, quaternion)

    invalid_sequence = [4, 5, 6, 7]

    with pytest.raises(ValueError):
        result = transform * invalid_sequence


def test_large_position_transform():
    large_position = [1e6, 1e6, 1e6]
    identity_rotation = Rotation.identity()
    transform = Transform(large_position, identity_rotation)
    large_position_transform = Transform(-np.array(large_position), identity_rotation)
    assert Transform.almost_equal(transform.inverse, large_position_transform)

