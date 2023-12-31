from PIL import Image, ImageDraw

WIDTH_2IN = 413
HEIGHT_2IN = 579


def prune_photo(photo):
    width = photo.size[0]
    height = photo.size[1]
    rate = height / width

    if rate < (HEIGHT_2IN / WIDTH_2IN):
        x = (width - int(height / HEIGHT_2IN * WIDTH_2IN)) / 2
        y = 0
        cut_photo = photo.crop((x, y, x + (int(height / HEIGHT_2IN * WIDTH_2IN)), y + height))
        return cut_photo
    else:
        x = 0
        y = (height - int(width / WIDTH_2IN * HEIGHT_2IN)) / 2
        cut_photo = photo.crop((x, y, x + width, y + (int(width / WIDTH_2IN * HEIGHT_2IN))))
        return cut_photo


def main():
    photo = Image.open('/Users/cxq/Downloads/s1.jpeg')
    print("w:{}, h:{}".format(photo.size[0], photo.size[1]))
    cut_photo = prune_photo(photo)
    print("w:{}, h:{}".format(cut_photo.size[0], cut_photo.size[1]))
    cut_photo.save('/Users/cxq/Downloads/s2.jpeg')


if __name__ == '__main__':
    main()
