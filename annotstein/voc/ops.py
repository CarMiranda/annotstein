import pathlib
from typing import Dict, Any, List
import itertools
import xml.etree.ElementTree as ET
from datetime import datetime


class VOC:
    @staticmethod
    def parse_xml(
        xml_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        """"""

        images = {}
        annotations: Dict[str, List[Dict[str, Any]]] = {}
        categories = {}
        c_annotations = c_categories = 1
        for image_id, xml_file in enumerate(xml_dir.glob("*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.findtext("filename", "")
            if filename == "":
                print(f"{xml_file} XML annotation has no image filename. Skipping.")
                continue

            folder = pathlib.Path(root.findtext("folder", ""))
            qualified_name = str(folder / filename)

            if qualified_name not in images:
                size = root.find("size")
                if size is None:
                    print(f"{xml_file} XML annotation has no size.")
                    width = 0
                    height = 0
                else:
                    width = int(size.findtext("width", 0))
                    height = int(size.findtext("height", 0))

                images[qualified_name] = {
                    "id": image_id,
                    "filename": qualified_name,
                    "width": width,
                    "height": height,
                }
                annotations[qualified_name] = []

            for member in root.findall("object"):
                klass = member.findtext("name")
                bndbox = member.find("bndbox")
                if bndbox is None or klass is None:
                    print(f"{xml_file} has annotations without bounding boxes or classes.")
                    continue

                coords = [bndbox.findtext(c) for c in ["xmin", "ymin", "xmax", "ymax"]]

                if any([c is None for c in coords]):
                    continue

                if klass not in categories:
                    categories[klass] = {
                        "id": c_categories,
                        "name": klass,
                        "supercategory": "",
                    }
                    c_categories += 1

                xmin, ymin, xmax, ymax = map(lambda x: int(x), coords)  # type: ignore

                annotations[qualified_name].append(
                    {
                        "id": c_annotations,
                        "image_id": images[qualified_name]["id"],
                        "category_id": categories[klass]["id"],
                        "segmentation": [],
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    }
                )
                c_annotations += 1

        now = datetime.now()

        return {
            "info": {
                "description": "",
                "url": "",
                "version": "1.0.0",
                "year": now.year,
                "contributor": "",
                "date_created": str(now),
            },
            "images": list(images.values()),
            "annotations": list(itertools.chain(*annotations.values())),
            "categories": list(categories.values()),
        }
