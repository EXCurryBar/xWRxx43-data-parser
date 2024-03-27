import time
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from .utils import Config, NumpyArrayEncoder, default_kwargs


class DataProcess:
    @default_kwargs(remove_static_noise=False, write_file=False, file_name=None)
    def __init__(self, data, **kwargs):
        self.raw_data = data
        self.args = kwargs
        self.length_list = list()
        self.xs = list()
        self.ys = list()
        self.zs = list()
        self.vs = list()
        self.rs = list()
        self.angles = list()
        self.elevs = list()
        if self.args["write_file"]:
            self._wrote_flag_raw = True
            self._wrote_flag_processed = True
            if self.args["file_name"] is None:
                self._file_name = datetime.today().strftime("%Y-%m-%d-%H%M")
            else:
                self._file_name = self.args["file_name"]
            self._writer = open(f"./raw_file/{self._file_name}.json", 'a', encoding="UTF-8")
            self._processed_output = open(f"./output_file/{self._file_name}.json", 'a', encoding="UTF-8")

    def process_cluster(self, detected_object, thr=10, delay=10):
        data = dict()
        points = detected_object["3d_scatter"]
        if len(self.length_list) >= delay:  # delay x * 0.1 s
            self.xs = self.xs[self.length_list[0]:]
            self.ys = self.ys[self.length_list[0]:]
            self.zs = self.zs[self.length_list[0]:]
            self.vs = self.vs[self.length_list[0]:]
            self.rs = self.rs[self.length_list[0]:]
            self.angles = self.angles[self.length_list[0]:]
            self.elevs = self.elevs[self.length_list[0]:]
            self.length_list.pop(0)
        self.length_list.append(len(points["x"]))

        self.xs += list(points["x"])
        self.ys += list(points["y"])
        self.zs += list(points["z"])
        self.vs += list(points["v"])
        self.rs += list(points["r"])
        self.angles += list(points["angle"])
        self.elevs += list(points["elev"])
        scatter_data = np.array([item for item in zip(self.xs, self.ys, self.zs)])
        if len(self.xs) > thr:
            try:
                z = linkage(scatter_data, method="ward", metric="euclidean")
            except:
                return 'r', [], [], []
            clusters = fcluster(z, 3.0, criterion='distance')
            color = list(clusters)
            labels = set(color)
            bounding_boxes = list()
            groups = list()
            new_groups = list()
            eigenvectors = list()
            for label in labels:
                xs = list()
                ys = list()
                zs = list()
                vs = list()
                rs = list()
                angles = list()
                elevs = list()
                if color.count(label) < thr:
                    outlier_index = [i for i in range(len(self.xs)) if color[i] == label]
                    for index in sorted(outlier_index, reverse=True):
                        del self.xs[index]
                        del self.ys[index]
                        del self.zs[index]
                        del self.vs[index]
                        del self.rs[index]
                        del self.angles[index]
                        del self.elevs[index]
                        del color[index]
                    continue

                for i in range(len(self.xs)):
                    if color[i] == label:
                        xs.append(self.xs[i])
                        ys.append(self.ys[i])
                        zs.append(self.zs[i])
                        vs.append(self.vs[i])
                        rs.append(self.rs[i])
                        angles.append(self.angles[i])
                        elevs.append(self.elevs[i])
                x1 = -999
                y1 = -999
                z1 = -999

                x2 = 999
                y2 = 999
                z2 = 999
                # group = [xs, ys, zs, vs]
                group = list()
                for idx, value in enumerate(color):
                    if value == label:
                        x, y, z = scatter_data[idx][:3]
                        x1 = x if x > x1 else x1
                        y1 = y if y > y1 else y1
                        z1 = z if z > z1 else z1

                        x2 = x if x2 > x else x2
                        y2 = y if y2 > y else y2
                        z2 = z if x2 > z else z2

                        group.append(scatter_data[idx][:3])
                        # print(x1, y1, z1)
                        # print(x2, y2, z2)
                groups.append(group)
                pca = PCA(n_components=3)
                eigenvalues = list()
                for group in groups:
                    new_group = pca.fit_transform(group)
                    # n_samples = np.shape(new_group[0])
                    new_group -= np.mean(new_group, axis=0)
                    new_groups.append(new_group)
                    # cov_matrix = np.dot(new_group.T, new_group) / n_samples
                    length = list()
                    vectors = pca.components_
                    eigenvalue = list(pca.explained_variance_)
                    eigenvalues.append(eigenvalue)
                    for eigenvector in vectors:
                        x, y, z = eigenvector
                        length.append(x ** 2 + y ** 2 + z ** 2)
                    # print(length.index(max(length)))
                    # print(values.index(max(values)))
                    eigenvectors.append(vectors[length.index(max(length))])

                    # print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
                z2 = -Config.RADAR_HEIGHT_IN_METER if z2 == 999 else z2
                bounding_boxes.append(
                    [
                        [[x1, y1, z1], [x1, y2, z1]],
                        [[x1, y1, z1], [x2, y1, z1]],
                        [[x1, y1, z1], [x1, y1, z2]],
                        [[x1, y1, z2], [x1, y2, z2]],
                        [[x1, y1, z2], [x2, y1, z2]],
                        [[x2, y1, z2], [x2, y2, z2]],
                        [[x2, y1, z2], [x2, y1, z1]],
                        [[x2, y1, z1], [x2, y2, z1]],
                        [[x2, y2, z2], [x2, y2, z1]],
                        [[x2, y2, z2], [x1, y2, z2]],
                        [[x1, y2, z2], [x1, y2, z1]],
                        [[x1, y2, z1], [x2, y2, z1]]
                    ]
                )
                data.update({
                    "scatter": points,
                    "bounding_box": bounding_boxes,
                    "group": groups,
                    "label": color,
                    "vector": eigenvectors,
                    "eigenvalues": eigenvalues
                })
                new_groups = self.project_on_plane(data)
                data["group"] = new_groups
            if self.args["write_file"]:
                self.write_processed_output(data)
            return color, groups, bounding_boxes, eigenvectors
        return 'r', [], [], []

    @staticmethod
    def project_on_plane(data):
        vectors = data["vector"]
        groups = data["group"]
        projected_group = list()
        z_vector = [0, 0, Config.RADAR_HEIGHT_IN_METER]
        for v, g in zip(vectors, groups):
            new_group = list()
            normal_vector = np.cross(z_vector, v[:2])
            for p in g:
                product = (-normal_vector[0] * p[0] - normal_vector[1] * p[1]) / (
                        normal_vector[0] ** 2 + normal_vector[1] ** 2)
                x_hat = p[0] - normal_vector[0] * product
                y_hat = p[1] - normal_vector[1] * product
                new_group.append([x_hat, y_hat])
            projected_group.append(new_group)
            # print(projected_group)
        return projected_group

    def write_processed_output(self, radar_data: dict):
        new_line = json.dumps(radar_data, cls=NumpyArrayEncoder)
        if self._wrote_flag_processed:
            self._processed_output.write(f"[[{time.time()}, {new_line}]")
            self._wrote_flag_processed = False
        else:
            self._processed_output.write(f",\n[{time.time()}, {new_line}]")

    def write_to_json(self, radar_data: dict):
        new_line = json.dumps(radar_data, cls=NumpyArrayEncoder)
        if self._wrote_flag_raw:
            self._writer.write(f"[[{time.time()}, {new_line}]")
            self._wrote_flag_raw = False
        else:
            self._writer.write(f",\n[{time.time()}, {new_line}]")