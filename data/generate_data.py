import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import cv2


def construct_rico_auto_encoder_data():
    """Compile the different data sources into a single file for easy manipulation using pandas."""
    data_directory = Path(__file__).parent
    ui_layout_vectors_path =  data_directory / "ui_layout_vectors"
    semantic_annotations_path = data_directory / "semantic_annotations"
    ui_data_path = data_directory / "combined"

    with open(ui_layout_vectors_path / "ui_names.json") as f:
        ui_layout_vector_names = json.load(f)["ui_names"]
        ui_layout_vector_names = zip(ui_layout_vector_names, range(len(ui_layout_vector_names)))

    ui_layout_vectors = np.load(ui_layout_vectors_path / "ui_vectors.npy")

    # print(len(ui_layout_vectors), len(list(semantic_annotations_path.glob("*.png"))), len(list(ui_data_path.glob("*.jpg"))))

    data = []
    for ui_img in tqdm(list(ui_data_path.glob("*.jpg")): #[:1000]):
        # if not (ui_data_path / (Path(name).stem + ".jpg")).exists():
        name = ui_img.stem
        semantic_img = semantic_annotations_path / (Path(name).stem + ".png")
        if not semantic_img.exists():
            print(semantic_img)
            raise Exception()
        entry = {"screenshot": str(ui_img), "semantic_img": str(semantic_img)}
        with open(ui_data_path / (name + ".json")) as f:
            ui_data = json.load(f)
            ui_leaf_views, c = _get_leaf_views(ui_data["activity"]["root"], [], 0)
            # img = cv2.imread(str(ui_img))
            
            # print(c, len(ui_leaf_views), img.shape, "*"*100)
            # h_factor = img.shape[0] / 2560
            # w_factor = img.shape[1] / 1440
            # for view in ui_leaf_views:
            #     if not view["visibility"] == "visible":
            #         color = (255, 0, 0)
            #     else:
            #         color = (255, 0, 255)
            #     x1, y1, x2, y2 = (np.array(view["bounds"]) * [w_factor, h_factor, w_factor, h_factor]).round().astype(np.int)
            #     print(x1, y1, x2, y2)
            #     print(view["focusable"])
            #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
            #     cv2.imshow("", img)
            #     cv2.waitKey()

            entry["ui_leaf_views"] = ui_leaf_views
            entry["ui_data"] = ui_data

        with open(semantic_annotations_path / (name + ".json")) as f:
            entry["semantic_data"] = json.load(f)

        data.append(entry)

    data_dir = data_directory / "generated"
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
    with open(data_dir / "rico.json", "w") as f:
        json.dump(data, f, indent=2)


def _get_leaf_views(el, leaf_els, total_count):
    total_count += 1
    if el is None:
        return leaf_els, total_count

    if "children" not in el or  len(el["children"]) == 0:
        leaf_els.append(el)
        return leaf_els, total_count

    for child_el in el["children"]:
        leaf_els, total_count = _get_leaf_views(child_el, leaf_els, total_count)

    return leaf_els, total_count

        
if __name__ == "__main__":
    construct_rico_auto_encoder_data()
