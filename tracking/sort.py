import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1]) +
                (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1]) - wh + 1e-6)
    return iou

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # NaN 방지
    if not np.isfinite(iou_matrix).all():
        iou_matrix = np.nan_to_num(iou_matrix)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices).T

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.tolist())

    return matches, unmatched_detections, unmatched_trackers

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.
        y = y1 + h / 2.
        s = w * h
        r = w / float(h)

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = np.array([[x], [y], [s], [r]])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.
        y = y1 + h / 2.
        s = w * h
        r = w / float(h)
        z = np.array([[x], [y], [s], [r]])
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x

    def get_state(self):
        x, y, s, r = self.kf.x[:4].flatten()

        # 안정성 보완: 음수 sqrt 방지
        if s <= 0 or r <= 0:
            w = h = 0
        else:
            w = np.sqrt(s * r)
            h = s / w
        return [x - w/2, y - h/2, x + w/2, y + h/2]

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0, 4))):
        trks = []
        for t in self.trackers:
            t.predict()
            trks.append(t.get_state())

        trks = np.array(trks)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for t, trk_idx in matched:
            self.trackers[trk_idx].update(dets[t])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i]))

        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        ret = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                ret.append(t.get_state() + [t.id])
        return np.array(ret)
