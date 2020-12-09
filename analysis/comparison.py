import numpy as np
import json
from pathlib import Path
from sklearn.neighbors import BallTree
from easydict import EasyDict
import cv2

def main():
    export_roots = list(Path("../exports").glob("*flat"))
    ui_layout_vector = get_ui_layout()

    for root_dir in (export_roots[3], ):
        train_data = get_trained_data(root_dir)
        for name, idx in list(train_data.name_to_idx.items())[:10]:
            train_d, train_i = train_data.tree.query([train_data.vectors[idx]], k=10)
            train_results = [train_data.idx_to_name[i] for i in train_i[0]]
            
            ui_idx = ui_layout_vector.name_to_idx[name]
            ui_d, ui_i= ui_layout_vector.tree.query([ui_layout_vector.vectors[ui_idx]], k=10)
            ui_results = [ui_layout_vector.idx_to_name[i] for i in ui_i[0]]

            print("ui")
            _display_images(name, *ui_results)
            print("train")
            _display_images(name, *train_results)

            print(train_results)
            print(ui_results)
        #     break
        # break
    
    # for name, idx in ui_layout_vector.name_to_idx:
    #     print(name, idx)
    #     break

    
def get_ui_layout():
    data_directory = Path("../data")

    ui_layout_vectors_path =  data_directory / "ui_layout_vectors"
    
    with open(ui_layout_vectors_path / "ui_names.json") as f:
        ui_layout_vector_names = [n.rstrip(".png") for n in json.load(f)["ui_names"]]
        ui_layout_vector_name_to_idx = zip(ui_layout_vector_names, range(len(ui_layout_vector_names)))
        ui_layout_vector_idx_to_name = zip(range(len(ui_layout_vector_names)), ui_layout_vector_names)

    ui_layout_vectors = np.load(ui_layout_vectors_path / "ui_vectors.npy")

    print(ui_layout_vectors.shape)
    ui_layout_vector_tree = BallTree(ui_layout_vectors)
    
    return Data(ui_layout_vector_tree, ui_layout_vector_name_to_idx, ui_layout_vector_idx_to_name, ui_layout_vectors)


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
    print(tree.query(encoding_vectors[:2]))

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
