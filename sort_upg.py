import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Vectorized IoU computation
def iou_matrix(dets, trks):
    if dets.size == 0 or trks.size == 0:
        return np.zeros((dets.shape[0], trks.shape[0]), dtype=np.float32)
    xA = np.maximum(dets[:,0,None], trks[None,:,0])
    yA = np.maximum(dets[:,1,None], trks[None,:,1])
    xB = np.minimum(dets[:,2,None], trks[None,:,2])
    yB = np.minimum(dets[:,3,None], trks[None,:,3])
    interW = np.clip(xB - xA, 0, None)
    interH = np.clip(yB - yA, 0, None)
    interArea = interW * interH
    areaD = ((dets[:,2] - dets[:,0]) * (dets[:,3] - dets[:,1]))[:,None]
    areaT = ((trks[:,2] - trks[:,0]) * (trks[:,3] - trks[:,1]))[None,:]
    union = areaD + areaT - interArea + 1e-6
    return interArea / union

# Optimized association using vectorized IoU
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if trackers.shape[0] == 0:
        return [], list(range(detections.shape[0])), []
    iou_mat = iou_matrix(detections, trackers)
    matched_indices = np.array(linear_sum_assignment(-iou_mat)).T
    unmatched_dets = [d for d in range(detections.shape[0]) if d not in matched_indices[:,0]]
    unmatched_trks = [t for t in range(trackers.shape[0]) if t not in matched_indices[:,1]]
    matches = []
    for d, t in matched_indices:
        if iou_mat[d, t] < iou_threshold:
            unmatched_dets.append(d)
            unmatched_trks.append(t)
        else:
            matches.append([d, t])
    return matches, unmatched_dets, unmatched_trks

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
        if s <= 0 or r <= 0:
            return [0,0,0,0]
        w = np.sqrt(s * r)
        h = s / w
        return [x - w/2, y - h/2, x + w/2, y + h/2]

class Sort:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0, 4))):
        # Predict new states for all trackers
        trks = []
        for trk in self.trackers:
            trk.predict()
            trks.append(trk.get_state())
        trks = np.array(trks)

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # Update matched trackers
        for d, t in matched:
            self.trackers[t].update(dets[d])

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i]))

        # Remove dead tracklets
        self.trackers = [trk for trk in self.trackers if trk.time_since_update < self.max_age]

        # Prepare output
        ret = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or trk.time_since_update == 0:
                ret.append(trk.get_state() + [trk.id])
        return np.array(ret)
