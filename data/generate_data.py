import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import multiprocessing

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

    # data = []
    # for ui_img in tqdm(list(ui_data_path.glob("*.jpg"))):  # [:1000]):
    #     # if not (ui_data_path / (Path(name).stem + ".jpg")).exists():

    #     data.append(entry)

    with multiprocessing.Pool(20) as p:
        ui_data_elements = list(ui_data_path.glob("*.jpg"))[:1000]
        data = []
        color = set([])
        for entry, c in tqdm(p.imap_unordered(ProcessEntry(semantic_annotations_path, ui_data_path), ui_data_elements, chunksize=1), total=len(ui_data_elements)):
            data.append(entry)
            # for cn in set(list(color.keys())).intersection(list(c.keys())):
            #     assert color[cn] == c[cn], str(color[cn]) + "  " + str(c[cn]) + f"     {cn}"
            # color.update(c)
            for cn in c.keys():
                color.add(cn)

    data_dir = data_directory / "generated"
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
    with open(data_dir / "rico.json", "w") as f:
        json.dump(data, f) # , indent=2)

    with open(data_dir / "semnantic_colors.json", "w") as f:
        out_color = {}
        for c, i in color:
            try:
                out_color[c].add(i)
            except KeyError:
                out_color[c] = set()
                out_color[c].add(i)
        for c in out_color.keys():
            out_color[c] = list(out_color[c])
            print(c, len(out_color[c]))
        print(len(out_color))
        json.dump(out_color, f, indent=2)


class ProcessEntry:
    def __init__(self, semantic_annotations_path, ui_data_path):
        self.semantic_annotations_path = semantic_annotations_path
        self.ui_data_path = ui_data_path
        
    def __call__(self, ui_img):
        name = ui_img.stem
        semantic_img = self.semantic_annotations_path / (Path(name).stem + ".png")
        if not semantic_img.exists():
            print(semantic_img)
            raise Exception()
        entry = {"name": name, "screenshot": str(ui_img), "semantic_img": str(semantic_img)}
        with open(self.ui_data_path / (name + ".json")) as f:
            ui_data = json.load(f)
            ui_leaf_views, c = _get_leaf_views(ui_data["activity"]["root"], [], 0)
            # img = cv2.imread(str(ui_img))

            # # print(c, len(ui_leaf_views), img.shape, "*"*100)
            # h_factor = img.shape[0] / 2560
            # w_factor = img.shape[1] / 1440
            # for view in ui_leaf_views:
            #     x1, y1, x2, y2 = (np.array(view["bounds"]) * [w_factor, h_factor, w_factor, h_factor]).round().astype(np.int)
            #     if "Text" in view["class"]: # view["visibility"] == "visible":
            #         color = (255, 0, 0)
            #     else:
            #         color = (0, 0, 255)

            #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
            #     print(x1, y1, x2, y2)
            #     print(view["class"], "----" if "text" not in view else view["text"])
                
            # cv2.imshow("", img)
            # cv2.waitKey()

            # entry["ui_leaf_views"] = ui_leaf_views
            # entry["ui_data"] = ui_data
            ui_leaf_views_filtered = []
            for view in ui_leaf_views:
                ui_leaf_views_filtered.append({"bounds": view["bounds"],
                                               "class": view["class"]})
            entry["ui_leaf_views"] = ui_leaf_views_filtered

        with open(self.semantic_annotations_path / (name + ".json")) as f:
            file_content = json.load(f)
            entry["semantic_data"], _, colors = _get_component_views(file_content, [], 0, {}, str(self.semantic_annotations_path / (name + ".png")))
            # print(json.dumps(entry["semantic_data"], indent=2), len(entry["semantic_data"]))
        return entry, colors


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


def _get_component_views(el, component_els, total_count, colors, img_path):
    total_count += 1
    if el is None:
        return component_els, total_count, colors

    if "componentLabel" in el:
        _el = el.copy()
        if "children" in _el:
            del _el["children"]
        component_els.append(_el)
        if el["componentLabel"] not in colors:
            component_name = (el["componentLabel"], None)
            if component_name[0] == "Icon":
                component_name = ("Icon",  el["iconClass"])
            elif component_name[0] == "Text Button":
                if "textButtonClass" not in el:
                    component_name = ("Text Button", "misc_text")
                else:
                    component_name = ("Text Button", el["textButtonClass"])
                
            # im = cv2.resize(cv2.imread(img_path), (1440, 2560))
            # y, x = el["bounds"][1] + 3, el["bounds"][0] + 3
            colors[component_name] = None #  im[y, x].tolist()
            

    if "children" in el:
        for child_el in el["children"]:
            component_els, total_count, colors = _get_component_views(child_el, component_els, total_count, colors, img_path)

    return component_els, total_count, colors

        
if __name__ == "__main__":
    construct_rico_auto_encoder_data()
