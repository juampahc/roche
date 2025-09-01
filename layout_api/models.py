import pymupdf

class GeometricParser(object):
    """
    This class is useful to split the page in non overlapping areas 
    from a region given a set of bboxes. The result cannot be contained
    in this set of bboxes.
    """
    def __init__(self, bboxes:list[pymupdf.Rect]):
        """
        For the constructor it is very important to check if the given
        bboxes do NOT overlap with each other. If that is the case
        then we will not be able to find non overlapping regions.
        """
        self.bboxes = bboxes
        self.word_bboxes = []
        
        pass
    
    def is_empty(self):
        """
        Return True if no word_bboxes else False
        """
        if len(self.word_bboxes)==0:
            return True
        else:
            return False
    
    def append(self, word:pymupdf.Rect) -> bool:
        """
        Method to insert a word in the list of results.
        It returns True or False depending on success.
        """
        # check if the word is contained in any of the bboxes
        if not any([bbox.intersects(word) for bbox in self.bboxes]):
            self.word_bboxes.append(word)
            return True
        else:
            return False
    
    def generate_regions(self) -> list[pymupdf.Rect]:
        """
        Generate a list of regions which are not contained in the bboxes.
        This regions are the sum of the word-bboxes.
        """
        if not self.word_bboxes:
            raise ValueError('No results to generate regions')
        regions = []
        for word_bbox in self.word_bboxes:
            # If there are no regions, add the result
            if not regions:
                regions.append(word_bbox)
            # If we have regions we need to check where to add
            else:
                new_region = False # flag
                for index in range(len(regions)):
                    # First add this bbox to the region
                    candidate = regions[index] | word_bbox
                    # The new region can NOT be contained in the bboxes
                    if not any([bbox.intersects(candidate) for bbox in self.bboxes]):
                        regions[index] = candidate
                        new_region = False
                    else:
                        new_region = True # change the flag
                # If the flag is true, it implies that it has remained
                # true after checking all the regions. Therefore this
                # word-bbox constitue a new region
                if new_region:
                    regions.append(word_bbox)
        return regions
                
        