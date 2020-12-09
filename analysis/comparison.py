import numpy as np
import json
from pathlib import Path
from sklearn.neighbors import BallTree
from sklearn.metrics import dcg_score
from easydict import EasyDict
import cv2
from mlpipeline import MetricContainer, iterator
from mlpipeline.utils import set_logger
from tqdm import tqdm
from repcomp.comparison import CCAComparison, NeighborsComparison
import multiprocessing

set_logger()

export_dirs = [
    "linear-flat",
    "linear-non-flat",
    "conv-flat",
    # "conv-non-flat",
]

def main():
    export_roots = [Path("../exports") / d for d in export_dirs]
    
    for root_dir in export_roots:  # (export_roots[2], ):
        train_data = get_trained_data(root_dir)
        ui_layout_vector = get_ui_layout(train_data)
        mc = MetricContainer(["precision", "recall", "dcg", "skipped"])
        # comparator = CCAComparison()
        # sim = comparator.run_comparison(train_data.vectors[:50], ui_layout_vector.vectors[:50])
        # print(sim)
        if "conv" in str(root_dir):
            n_proc = 2
        else:
            n_proc = 7
        with multiprocessing.Pool(n_proc) as p:
            if "conv" in str(root_dir):
                map_fn = lambda p, i: map(p, i)
            else:
                map_fn = lambda pr, i: p.imap_unordered(pr, i, chunksize=100)

            print(map_fn, n_proc)
            for out in tqdm(map_fn(_process(train_data, ui_layout_vector), iterator(train_data.name_to_idx.items(), None)), total=len(train_data.name_to_idx)):
                precision, recall, dcg, skipped = out
                if skipped:
                    mc.skipped.update(1)
                    continue
                else:
                    mc.skipped.update(0)
                    mc.precision.update(precision)
                    mc.recall.update(recall)
                    mc.dcg.update(dcg)

        #     break
        # break
        mc.log_metrics()
    
    # for name, idx in ui_layout_vector.name_to_idx:
    #     print(name, idx)
    #     break



class _process:
    def __init__(self, train_data, ui_layout_vector):
        self.train_data = train_data
        self.ui_layout_vector = ui_layout_vector

    def __call__(self, x):
        name, idx = x
        train_d, train_i = self.train_data.tree.query([self.train_data.vectors[idx]], k=20)
        train_results = [self.train_data.idx_to_name[i] for i in train_i[0]]

        try:
            ui_idx = self.ui_layout_vector.name_to_idx[name]
            ui_d, ui_i= self.ui_layout_vector.tree.query([self.ui_layout_vector.vectors[ui_idx]], k=20)
            ui_results = [self.ui_layout_vector.idx_to_name[i] for i in ui_i[0]]
            # print("ui")
            # _display_images(name, *ui_results)
            # print("train")
            # _display_images(name, *train_results)
            
            # print(train_results)
            # print(ui_results)
        except KeyError:
            return None, None, None, True

        return _precision(ui_results, train_results), _recall(ui_results, train_results), _dcg(ui_results, train_results), False

        
def _precision(ui_results, train_results):
    return len(set(train_results).intersection(ui_results))/len(train_results)


def _recall(ui_results, train_results):
    return len(set(train_results).intersection(ui_results))/len(ui_results)


def _dcg(ui_results, train_results):
    lables = set(ui_results)
    lables.update(train_results)
    true_labels = [[1 if l in ui_results else 0 for l in lables]]
    pred_labels = [[1 if l in train_results else 0 for l in lables]]
    return dcg_score(true_labels, pred_labels)


def get_ui_layout(train_data):
    data_directory = Path("../data")

    ui_layout_vectors_path =  data_directory / "ui_layout_vectors"
    
    with open(ui_layout_vectors_path / "ui_names.json") as f:
        names = [n.rstrip(".png") for n in json.load(f)["ui_names"]]
        name_to_idx = zip(names, range(len(names)))
        idx_to_name = zip(range(len(names)), names)

    vectors = np.load(ui_layout_vectors_path / "ui_vectors.npy")

    filtered_name_to_idx = []
    filtered_idx_to_name = []
    filtered_vectors = []
    for idx in range(len(train_data.vectors)):
        name = train_data.idx_to_name[idx]
        
        filtered_name_to_idx.append((name, idx))
        filtered_idx_to_name.append((idx, name))
        filtered_vectors.append(vectors[idx])

    filtered_vectors = np.array(filtered_vectors)
    # print(filtered_vectors.shape)
    tree = BallTree(filtered_vectors)
    
    return Data(tree, filtered_name_to_idx, filtered_idx_to_name, filtered_vectors)


def _strip_npy_name(name):
    return name[0].lstrip("combined/").rstrip(".jpg")


def get_trained_data(root_dir):
    train_encodings = dict([(_strip_npy_name(n), v) for n, v in np.load(root_dir / "train_encodings.npy", allow_pickle=True).item().items()])
    test_encodings = dict([(_strip_npy_name(n), v) for n, v in np.load(root_dir / "test_encodings.npy", allow_pickle=True).item().items()])
    assert len(set(list(train_encodings.keys())).intersection(list(test_encodings.keys()))) == 0

    train_names = list(train_encodings.keys())
    test_names = list(test_encodings.keys())

    encodings = train_encodings
    encodings.update(test_encodings)

    name_to_idx = []
    idx_to_name = []
    encoding_vectors = []

    for idx, (n, v) in enumerate(encodings.items()):
        name_to_idx.append([n, idx])
        idx_to_name.append([idx, n])
        encoding_vectors.append(v.flatten())

    encoding_vectors = np.array(encoding_vectors)
    print(idx, encoding_vectors.shape, root_dir)
    tree = BallTree(encoding_vectors)
    # print(tree.query(encoding_vectors[:2]))

    return Data(tree, name_to_idx, idx_to_name, encoding_vectors, train_names, test_names)


def _display_images(*img_names):
    imgs = []
    for img_name in img_names:
        img = cv2.imread(f"../data/combined/{img_name}.jpg")
        if img is None:
            img = np.ndarray((1440, 2560, 3), dtype=np.uint8)
        img = cv2.resize(img, (1440, 2560))
        imgs.append(img)

    imgs.append(np.ndarray((2560, 1440, 3), dtype=np.uint8))

    img = cv2.vconcat([cv2.hconcat(imgs[i: i+6]) for i in range(0, 12, 6)])
    # print([i.shape for i in [cv2.hconcat(imgs[i: i+6]) for i in range(0, 12, 6)]])

    cv2.imshow("", img)
    cv2.waitKey()


class Data:
    def __init__(self, tree, name_to_idx, idx_to_name, vectors, train_names=None, test_names=None):
        self.tree = tree
        self.name_to_idx = dict(name_to_idx)
        self.idx_to_name = dict(idx_to_name)
        self.vectors = vectors
        self.train_names = train_names
        self.test_names = test_names

if __name__ == "__main__":
    main()
