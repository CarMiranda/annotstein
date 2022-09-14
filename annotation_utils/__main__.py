import pathlib
import argparse

from annotation_utils.coco import COCO
from annotation_utils.voc import VOC


def get_args():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Rebase filename
    rebase_parser = subparsers.add_parser("rebase")
    rebase_parser.add_argument(
        "-i", "--input-files", dest="input_files", nargs="+", type=pathlib.Path
    )
    rebase_parser.add_argument("-p", "--prefixes", nargs="+", type=pathlib.Path)
    rebase_parser.add_argument("--inplace", default=False, action="store_true")

    # Merge multiple annotation files
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument(
        "-i", "--input-files", dest="input_files", nargs="+", type=pathlib.Path
    )
    merge_parser.add_argument("-o", "--output", type=pathlib.Path)

    # Convert from VOC to COCO
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--input", type=pathlib.Path)
    convert_parser.add_argument("-o", "--output", type=pathlib.Path)
    convert_parser.add_argument("-s", "--source", type=str, choices=["voc"])
    convert_parser.add_argument("-t", "--target", type=str, choices=["coco"])

    # Convert coordinates from relative to absolute
    coords_parser = subparsers.add_parser("coords")
    coords_parser.add_argument("--input", type=pathlib.Path)
    coords_parser.add_argument("-d", "--root-dir", type=pathlib.Path)
    coords_parser.add_argument("-o", "--output", type=pathlib.Path)
    coords_parser.add_argument("-t", "--target", type=str, choices=["abs", "rel"])

    # Extract crops for classification
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument(
        "-i", "--input-files", dest="input_files", nargs="+", type=pathlib.Path
    )
    extract_parser.add_argument(
        "-d", "--root-dirs", dest="root_dirs", nargs="+", type=pathlib.Path
    )
    extract_parser.add_argument(
        "-o", "--output-dir", dest="output_dir", type=pathlib.Path
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.command == "rebase":
        for index_path, prefix in zip(args.input_files, args.prefixes):
            if args.inplace:
                output_path = index_path
            else:
                output_path = index_path.parent / ("rebased-" + index_path.name)

            print("Writing to " + str(output_path))
            COCO.from_file(index_path).rebase_filenames(prefix).write(output_path)
    elif args.command == "merge":
        indices = []
        for index_path in args.input_files:
            indices.append(COCO.from_file(index_path))

        COCO.merge(*indices).write(args.output)
    elif args.command == "convert":
        source = args.source
        target = args.target

        if source == "voc" and target == "coco":
            voc = VOC.parse_xml(args.input)
            COCO.from_dict(voc).write(args.output)
    elif args.command == "coords":
        coco = COCO.from_file(args.input)
        if args.target == "rel":
            coco.to_relative_coordinates(args.root_dir).write(args.output)
        elif args.target == "abs":
            coco.to_absolute_coordinates(args.root_dir).write(args.output)
    elif args.command == "extract":
        for index_path, root_dir in zip(args.input_files, args.root_dirs):
            COCO.from_file(index_path).to_classtree(root_dir, args.output_dir)


if __name__ == "__main__":
    main()
