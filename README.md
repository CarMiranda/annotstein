# Annotstein

`annotstein` provides lightweight and modular endpoints to common annotation operations, like rebasing file paths, converting coordinates from absolute to relative, etc.

***NOTE: this library is unstable and under active development.***

## Library
Currently, the lib allows to manipulate [COCO](https://cocodataset.org/), [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [ultralytics YOLO](https://docs.ultralytics.com/datasets/detect/) datasets through object-oriented and functional APIs.
The COCO format is used as the exchange format to manipulate annotations, i.e. most operations on VOC and YOLO datasets transform the dataset to COCO as an intermediate, in-memory step.

These APIs allow to perform multiple operations on a dataset, like converting from one format to another, merge multiple datasets, extract crops based on bboxes, etc.

Classes under `annotstein.{coco,voc,yolo}.schemas` provide validation using [pydantic](https://docs.pydantic.dev/latest/) models. These classes can be sub-classed to add extra-validation (e.g. specific categories are expected) and still work with the functional APIs, see [examples/subclassing.py](https://github.com/CarMiranda/annotstein/blob/main/examples/subclassing.py).

## CLI

### Convert
The `convert` command allows to convert between annotation formats.

```bash
# Convert VOC to COCO
convert -s voc -t coco -i voc/ -o coco.json

# Convert COCO to VOC
convert -s coco -t voc -i coco.json -o voc/

# Convert YOLO-segment (https://docs.ultralytics.com/datasets/segment/) to COCO
convert -s yolo-segment-task -t coco -i dataset.yaml -o coco.json

# Convert COCO to YOLO-segment
convert -s coco -t yolo-segment-task -i coco.json -t dataset.yaml

# Convert YOLO-detect (https://docs.ultralytics.com/datasets/detect/) to COCO
convert -s yolo-detect-task -t coco -i dataset.yaml -o coco.json

# Convert COCO to YOLO-detect
convert -s coco -t yolo-detect-task -i coco.json -o dataset.yaml
```

Since VOC and YOLO require annotations to exist as individual text files along with the images they describe, the flags `--symlink|--copy|--move` in the `convert` command allow to select how images will be processed.

### COCO operations

The following operations can be performed on COCO datasets, using the `coco` subcommand:
- Transform coordinates from absolute to relative, and vice verse (`coordinates`)
- Merge multiple COCO datasets into a single one (`merge`)
- Extract image patches into a `category_id`-based file tree (`crop`)
- Rebase a given dataset images' `file_name`s (`rebase`)
- Compute some stats (e.g. category instance count) (`stats`)
- Split a COCO dataset into train and test datasets (`split`)

## Uses

The aim of this project is to allow simple manipulation of CV datasets through the API or the CLI. Moreover, it can also be used in CI/CD or applications steps to validate that annotations are coherent (e.g. validate COCO dataset's bboxes and segmentations, validate annotations refer to existing images, etc).

You can obviously use it for whatever you want!

## Roadmap
- Integration with CVOps tooling (e.g. [CVAT](https://www.cvat.ai/), [FiftyOne](https://docs.voxel51.com/))
- Add missing operations for YOLO datasets
- Add more utility functions for COCO datasets

## References
- [COCO](https://cocodataset.org/) challenge and dataset
- [Pascal-VOC](http://host.robots.ox.ac.uk/pascal/VOC/) challenge and dataset
- Ultralytics [YOLO-detect](https://docs.ultralytics.com/datasets/detect/) and [YOLO-segment](https://docs.ultralytics.com/datasets/segment/) ~~hidden~~ docs
