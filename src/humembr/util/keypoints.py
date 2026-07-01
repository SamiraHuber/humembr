def keypoint_vis_score(keypoints: list[tuple[float, float, float]], confidence=0.75):
    assert len(keypoints) == 17, f"Expected 18 keypoints, got {len(keypoints)}"
    all_vis = []
    for k in keypoints:
        _, _, vis = k
        if vis >= confidence:
            all_vis.append(1)
    return sum(all_vis) / 17
