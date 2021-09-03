Feature Generation CLIs
=======================

Luna Pathology feature generation CLIs are installed as binaries as part of the installation.
Please see the documentation of the CLIs on this page.
For more details, please checkout our <link-to-tutorials>.

Tiling CLIs
-----------

As a whole slide image is too large for deep learning model training, a slide is often divded into a set of small tiles, and used for training.
For tile-based whole slide image analysis, generating tiles and labels is an important and laborious step.
With Luna tiling CLIs and tutorials, you can easily generate tile labels and get your data ready for downstream analysis.

.. click:: luna.pathology.cli.load_slide:cli
   :prog: load_slide
   :nested: full

.. click:: luna.pathology.cli.generate_tile_labels:cli
   :prog: generate_tiles
   :nested: full

.. click:: luna.pathology.cli.collect_tile_segment:cli
   :prog: collect_tiles
   :nested: full

.. click:: luna.pathology.cli.infer_tile_labels:cli
   :prog: infer_tiles
   :nested: full

.. click:: luna.pathology.cli.visualize_tile_labels:cli
   :prog: visualize_tiles
   :nested: full

DSA CLIs
--------

Digital Slide Archive (DSA) is a platform where you can manage your pathology images and annotations.
DSA also provides APIs that we use to push model results to the platform.
A set of CLIs are available to help you convert your pathologist or model-generated annotations and push them to DSA.

.. click:: luna.pathology.cli.dsa.dsa_upload:cli
   :prog: dsa_upload
   :nested: full

.. click:: luna.pathology.cli.dsa.dsa_viz:cli
   :prog: dsa_viz
   :nested: full
