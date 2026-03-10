def landmark_to_pixel(landmark, frame_shape):
    h, w = frame_shape[:2]
    return int(landmark.x * w), int(landmark.y * h)


def bbox_contains_point(bbox, point, margin=0):
    """
    bbox: (x1, y1, x2, y2)
    point: (x, y)
    margin: tolerance in pixels
    """
    x1, y1, x2, y2 = bbox
    px, py = point

    return (
        x1 - margin <= px <= x2 + margin and
        y1 - margin <= py <= y2 + margin
    )
