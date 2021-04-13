import numpy as np
from parzen_dynamic.ValueBuilder import ValueBuilder


class GraphicBuilder:
    def __init__(self):
        self.WINDOW_WIDTH = 30 / 2.54
        self.WINDOW_HEIGHT = 13 / 2.54

        self.DOT_LEARNING_SIZE = 1
        self.DOT_LEARNING_COLOR = 'red'
        self.DOT_TESTING_SIZE = 25
        self.DOT_TESTING_COLOR = 'darkblue'

        self.LINE_WIDTH = 0.5
        self.LINE_COLOR = 'black'
        self.SELECTED_WIDTH = 2.0

        self.MIN_CLASSES = 1
        self.MAX_CLASSES = 15

        self.MIN_K = 1
        self.MAX_K = 25

        self.k = 4

        # x1
        self.a = 14.0
        self.b = 16.0

        # x2
        self.c = 2.0
        self.d = 4.0

        # n - x1 classes, m - x2 classes
        self.n = 2
        self.m = 2

        # amount of Learning Dots
        self.l = 500
        self.MIN_LEARNING_DOTS = 10
        self.MAX_LEARNING_DOTS = 2000

        # amount of nearest members
        # self.k = 4

        self.DISTANCE_ALGORITHM = ValueBuilder.euclidian_distance

        # self.WEIGHT_ALGORITHM = ValueBuilder.no_weigh
        self.KERNEL = ValueBuilder.boolean_kernel

        self.etalons = []
        self.etalonsPercentage = 0.3
        self.etalons_horizontal = []
        self.etalons_vertical = []

        self.ETALONS_SIZE = 8
        self.ETALONS_COLOR = 'magenta'

        # building points for segments equally
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)

        self.generate_dots_and_clasify()

        self.generate_and_clasify_dot()

        self.h = self.distances[self.k-1][2]

    def generate_dots_and_clasify(self):
        # generating learning dots
        self.learning_dots_horizontal = np.random.uniform(self.a, self.b, size=self.l)
        self.learning_dots_vertical = np.random.uniform(self.c, self.d, size=self.l)
        self.learning_dots = np.column_stack((self.learning_dots_horizontal, self.learning_dots_vertical))
        self.learning_dots = list(sorted(self.learning_dots, key=lambda x: (x[0], x[1])))
        # classifying dots
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def generate_ab_and_classify(self):
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def generate_cd_and_classify(self):
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def generate_and_clasify_dot(self):
        # Generating random point
        self.random_point = ValueBuilder.get_random_point(self.a, self.b, self.c, self.d)
        self.classify_dot()

    def getEtalons(self):
        result = list()
        values = set(map(lambda x: x[1], self.etalons))
        newlist = [[y for y in self.etalons if y[1] == x] for x in values]
        for dotsByClass in newlist:
            classSize = len(dotsByClass)
            etalonsNeeded = int(classSize * self.etalonsPercentage)
            for i in range(0, etalonsNeeded):
                result.append(dotsByClass[i])
                self.etalons_horizontal.append(dotsByClass[i][0][0])
                self.etalons_vertical.append(dotsByClass[i][0][1])
        return result

    def classify_dot(self):
        # findEtalons() : sorted by ASC array(coords, class, sum of distances to 'classmates')
        self.etalons = ValueBuilder.findEtalons(self.classified_dots)

        self.etalonsChosen = self.getEtalons()

        # array(coords,class,distance to random point)
        self.distances = ValueBuilder.distance_to_dots(self.random_point, self.etalonsChosen,
                                                       self.DISTANCE_ALGORITHM)

        self.h = self.distances[self.k-1][2]

        self.point_class = ValueBuilder.classify_by_parzen_dynamic(self.distances, self.KERNEL, self.h)

    def get_hor_ver_class(self):
        h = 0
        while self.point_class - h * self.n > 0:
            h += 1
        h = h - 1 if h > 0 else 0
        v = self.point_class - h * self.n - 1
        return h, v

    def box_hor_ver(self):
        h, v = self.get_hor_ver_class()
        hor_res = [h, h + 1]
        ver_res = [v, v + 1]

        return hor_res, ver_res
