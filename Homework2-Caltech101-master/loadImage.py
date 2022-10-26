from PIL import Image
class LoadImage:


    def __init__(self, root,label, imagePath):
        self.label = label
        # self.imagePath = root+imagePath
        self.typePIL = self.pil_loader( root + imagePath )

    def __repr__(self):
        return f'label= {self.label}'

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')