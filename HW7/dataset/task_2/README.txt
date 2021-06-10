========================================================
CelebA-HQ Dataset
========================================================

CelebA-HQ is a large-scale face image dataset that has 30,000 high-resolution face images selected from the CelebA dataset. Each image has facial attributes corresponding to CelebA.

You can find images in CelebA-HQ-img, and attribute annotation for each image in CelebAMask-HQ-attribute-anno.txt.

CelebA-HQ-attribute-anno.txt has 3 parts:
    * The very first line in CelebA-HQ-attribute-anno.txt is the number of total images, that is, 30000.
    * The second line shows 40 attributes. The order of clasess correspond to label order in below lines
    * There will be 30000 remaining lines. Each line represents 1 image name and the attribute annotations.
      For example, the third line in CelebA-HQ-attribute-anno.txt is:
        `0.jpg  -1 1 1 1 -1 -1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 1 1 -1 1 -1 -1 1 -1 -1 1 -1 -1 -1 1 -1 1 -1 -1 1 -1 -1 1`
        It shows that image 0.jpg does not have 1st attribute, but has 2nd, 3rd, 4th attributes in it, and so on.


For more information about the dataset, visit the project website:

  https://github.com/switchablenorms/CelebAMask-HQ

