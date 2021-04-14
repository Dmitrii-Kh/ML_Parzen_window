import numpy as np


class ValueBuilder:

    @staticmethod
    def spread_segment_equally(x1, x2, n):
        return np.linspace(x1, x2, num=n)

    @staticmethod
    def classify_dots(ab_segment, cd_segment, learning_dots):
        result = list()
        for learning_dot in learning_dots:
            # x1 - x, x2 - y
            for i1, x1 in enumerate(ab_segment):
                if learning_dot[0] < x1:
                    for i2, x2 in enumerate(cd_segment):
                        if learning_dot[1] < x2:
                            result.append([learning_dot, i1 + (i2 - 1) * (len(ab_segment) - 1)])
                            break
                    break
        return result

    @staticmethod
    def euclidian_distance(dot1, dot2):
        return np.sqrt(np.power(dot1[0] - dot2[0], 2) + np.power(dot1[1] - dot2[1], 2))

    @staticmethod
    def manhattan_distance(dot1, dot2):
        return np.absolute(dot1[0] - dot2[0]) + np.absolute(dot1[1] - dot2[1])

    @staticmethod
    def boolean_kernel(r):
        return 1 if abs(r) <= 1 else 0

    @staticmethod
    def linear_kernel(r):
        return 1 - abs(r) if abs(r) <= 1 else 0

    @staticmethod
    def quadratic_kernel(r):
        return 1 - r * r if abs(r) <= 1 else 0

    @staticmethod
    def gauss_kernel(r):
        return 2.718 ** (-2 * r * r)

    @staticmethod
    def getClassmates(dotClass, dots):
        return filter(lambda x: x[1] == dotClass, dots)

    @staticmethod
    def findEtalons(dots):
        result = list()
        for dot in dots:
            classmates = ValueBuilder.getClassmates(dot[1], dots)
            distanceToClassmates = 0
            for classmate in classmates:
                distanceToClassmates += ValueBuilder.euclidian_distance(dot[0], classmate[0])
            result.append([dot[0], dot[1], distanceToClassmates])

        return list(sorted(result, key=lambda x: x[2]))

    @staticmethod
    def getShortestLengthToEtalon(dot, etalons):
        minLength = 1000
        for etalon in etalons:
            currLength = ValueBuilder.euclidian_distance(dot, etalon[0])
            if minLength > currLength:
                minLength = currLength
        return minLength

    @staticmethod
    def distance_to_dots(dot, learning_dots, distance_method):
        result = list()
        for learning_dot in learning_dots:
            result.append([learning_dot[0], learning_dot[1], distance_method(dot, learning_dot[0])])
        return list(sorted(result, key=lambda x: x[2]))

    @staticmethod
    def get_random_point(a, b, c, d):
        return [np.random.uniform(a, b), np.random.uniform(c, d)]

    @staticmethod
    def classify_by_parzen_fixed(neighbours, kernel, h):
        result = dict()
        for index, neighbour in enumerate(neighbours):
            try:
                r = neighbour[2] / h
                result[neighbour[1]] += kernel(r)
            except KeyError:
                r = neighbour[2] / h
                result[neighbour[1]] = kernel(r)
        maxValClass = max(result, key=lambda key: result[key])
        return maxValClass if result[maxValClass] > 0 else -1

    @staticmethod
    def select_right_h(learning_dots, distance_method, kernel):
        result = dict()
        for h in np.arange(0.05, 1, 0.05):
            print(h)
            result[h] = ValueBuilder.classify_miss_by_h(learning_dots, distance_method, kernel, h)
        print(result)
        return max(result, key=lambda key: (result[key], key))

    @staticmethod
    def classify_miss_by_h(learning_dots, distance_method, kernel, h):
        result = 0
        for index, learning_dot in enumerate(learning_dots):
            testing_data = learning_dots[:index] + learning_dots[index + 1:]
            distances = ValueBuilder.distance_to_dots(learning_dot[0], testing_data, distance_method)
            test_class = ValueBuilder.classify_by_parzen_fixed(distances, kernel, h)
            if test_class == learning_dot[1]:
                result += 1
            else:
                continue
        return result
