import os
import glob
from video import video
from video import frame
import xml.etree.ElementTree as ET
from ..logger.logger import setup_logger
from src.helper.BoundingBox import BoundingBox

logger = setup_logger(logfile=None)

class loader_vid:
    """Docstring for vid2015. """

    def __init__(self, vid_folder, annotations_folder, logger):
        """TODO: to be defined1. """

        self.logger = logger
        self.vid_folder = vid_folder
        self.annotations_folder = annotations_folder
        self.vid_videos = {}
        self.category = {}
        if not os.path.isdir(vid_folder):
            logger.error('{} is not a valid directory'.format(vid_folder))

    def loaderVID(self):
        """TODO: Docstring for loaderVID.
        :returns: TODO

        """
        logger = self.logger
        vid_folder = self.vid_folder
        vid_subdirs = sorted(self.find_subfolders(self.annotations_folder))
        num_annotations = 0
        dict_list_of_annotations = {}

        for i, vid_sub_folder in enumerate(vid_subdirs):
            vid_folder_ann = sorted(self.find_subfolders(os.path.join(self.annotations_folder, vid_sub_folder)))

            logger.info(
                'Loading {:>3} of {:>3} - annotation file from folder = {:>4} has {:>3} videos'.format(i + 1,
                                                                                      len(vid_subdirs),
                                                                                      vid_sub_folder,
                                                                                      len(vid_folder_ann)))
            # vidoes
            for idx, ann in enumerate(vid_folder_ann):
                self.load_annotation_file(vid_sub_folder, ann)

    def find_subfolders(self, vid_folder):
        """TODO: Docstring for find_subfolders.

        :vid_folder: directory for vid videos
        :returns: list of video sub directories
        """

        return [dir_name for dir_name in os.listdir(vid_folder) if os.path.isdir(os.path.join(vid_folder, dir_name))]

    def load_annotation_file(self, vid_sub_folder, annotation_file):

        video_path = os.path.join(self.vid_folder, vid_sub_folder, annotation_file)

        objVideo = video(video_path)
        all_frames = glob.glob(os.path.join(video_path, '*.JPEG'))
        logger.info('{:>4} has {:>3} images'.format(annotation_file, len(all_frames)))
        objVideo.all_frames = sorted(all_frames)

        annotation_xml = sorted(glob.glob(os.path.join(self.annotations_folder, vid_sub_folder, annotation_file, '*.xml')))

        for frame_num, xml in enumerate(annotation_xml):
            root = ET.parse(xml).getroot()
            folder = root.find('folder').text
            filename = root.find('filename').text
            size = root.find('size')
            disp_width = int(size.find('width').text)
            disp_height = int(size.find('height').text)

            # trackid 0 only
            for obj in root.findall('object'):
                if obj.find('trackid').text == '0':
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymin = int(bbox.find('ymin').text)
                    ymax = int(bbox.find('ymax').text)

                    bbox = BoundingBox(xmin, ymin, xmax, ymax)
                    objFrame = frame(frame_num, bbox)
                    objVideo.annotations.append(objFrame)

        video_name = video_path.split('/')[-1]
        self.vid_videos[video_name] = objVideo
        if vid_sub_folder not in self.category.keys():
            self.category[vid_sub_folder] = []

        self.category[vid_sub_folder].append(self.vid_videos[video_name])


    def get_videos(self, isTrain=True, val_ratio=0.2):
        """TODO: Docstring for get_videos.
        :returns: TODO
        """

        videos = []
        logger = self.logger
        num_categories = len(self.category)
        category = self.category
        keys = sorted(category.keys())
        count = 0
        for i in range(num_categories):
            category_video = category[keys[i]]
            num_videos = len(category_video)
            num_val = int(val_ratio * num_videos)
            num_train = num_videos - num_val

            if isTrain:
                start_num = 0
                end_num = num_train - 1
            else:
                start_num = num_train
                end_num = num_videos - 1

            for i in range(start_num, end_num + 1):
                video = category_video[i]
                videos.append(video)

        num_annotations = 0
        for i, _ in enumerate(videos):
            num_annotations = num_annotations + len(videos[i].annotations)

        logger.info('Total annotated video frames: {}'.format(num_annotations))

        return videos


if '__main__' == __name__:
    logger = setup_logger(logfile=None)
    objLoaderVID = loader_vid('/home/jaehyuk/dataset/ILSVRC2015/images',
                                '/home/jaehyuk/dataset/ILSVRC2015/gt', logger)
    objLoaderVID.loaderVID()
    objLoaderVID.get_videos()
