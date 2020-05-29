
# Tests the SegmentDataset and ChainDataset classes
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../data")
    parser.add_argument("-segment_file", type=str, default="segments.json")
    parser.add_argument("-chains_file", type=str, default="val_chains.json")
    parser.add_argument("-vocab_file", type=str, default="vocab.csv")
    parser.add_argument("-vectors_file", type=str, default="vectors.json")
    parser.add_argument("-split", type=str, default="val")
    args = parser.parse_args()

    print("Loading the vocab...")
    vocab = Vocab(os.path.join(args.data_path, args.vocab_file), 3)

    print("Testing the SegmentDataset class initialization...")

    segment_val_set = SegmentDataset(
        data_dir=args.data_path,
        segment_file=args.segment_file,
        vectors_file=args.vectors_file,
        split=args.split
    )

    print("Testing the SegmentDataset class item getter...")
    print("Dataset contains {} segment samples".format(len(segment_val_set)))
    sample_id = 2
    sample = segment_val_set[sample_id]
    print("Segment {}:".format(sample_id))
    print("Image set: {}".format(sample["image_set"]))
    print("Target image index(es): {}".format(sample["targets"]))
    print("Target image Features: {}".format([segment_val_set.image_features[sample["image_set"][int(target)]] for target in sample["targets"]]))
    print("Encoded segment: {}".format(sample["segment"]))
    print("Decoded segment dialogue: {}".format(vocab.decode(sample["segment"])))
    print("Segment length: ", sample["length"])
    print("\nDone.")