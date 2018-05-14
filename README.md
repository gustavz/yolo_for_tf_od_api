# This is a collection of  all necessary files to include YoloV2 / Darknet to tensorflow's [object_detection repo](https://github.com/tensorflow/models/tree/master/research/object_detection)

> Note: The original files in the `tensorflow/models/research/object_detection` repo changed since this project (March 2018). Manual copying of the yolo related functions into the updated files of the repo may be necessary!

## Added or Updated file list
These files need to be copied into the following `tensorflow/models/` directories
> Hint: To be secure you may want to backup the original files that will be overwritten (although the new files do not change/delete existing code but only extend it)

    Config:
        research/object_detection/samples/configs/yolo_v2_darknet_19_voc.config

    Anchor box:
        research/object_detection/anchor_generators/yolo_grid_anchor_generator.py
        research/object_detection/anchor_generators/yolo_grid_anchor_generator_test.py
        research/object_detection/builders/anchor_generator_builder.py

    Loss:
        research/object_detection/core/losses.py
        research/object_detection/builders/losses_builder.py

    Feature map:
        research/slim/nets/darknet.py
        research/slim/nets/darknet_test.py
        research/object_detection/models/yolo_v2_darknet_19_feature_extractor.py
        research/object_detection/models/yolo_v2_darknet_19_feature_extractor_test.py

    Proto:
        research/object_detection/protos/anchor_generator.proto
        research/object_detection/protos/yolo_anchor_generator.proto
        research/object_detection/protos/losses.proto

    ETC:
        research/object_detection/builders/model_builder.py
        research/object_detection/meta_architectures/yolo_meta_arch.py

**©[rky0930](https://github.com/rky0930/models/blob/object_detection_yolo/research/object_detection/README.md)**
