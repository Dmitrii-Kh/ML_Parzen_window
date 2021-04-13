import numpy as np


class ValueBuilder:

    @staticmethod
    def getClassmates(dotClass, dots):
        return filter(lambda x: x[1] == dotClass, dots)

    # @staticmethod
    # def getClassmates(dotClass, dots):
    #     result = list()
    #     for dot in dots:
    #         if dot[1] == dotClass:
    #             result.append(dot)
    #     return result

    @staticmethod
    def findEtalons(dots):
        result = list()
        for dot in dots:
            classmates = ValueBuilder.getClassmates(dot[1], dots)
            # length = len(classmates)
            distanceToClassmates = 0
            for classmate in classmates:
                distanceToClassmates += ValueBuilder.euclidian_distance(dot[0], classmate[0])
            result.append([dot[0], dot[1], distanceToClassmates])

        return list(sorted(result, key=lambda x: x[2]))

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
    def distance_to_dots(dot, learning_dots, distance_method):
        result = list()
        for learning_dot in learning_dots:
            result.append([learning_dot[0], learning_dot[1], distance_method(dot, learning_dot[0])])
        return list(sorted(result, key=lambda x: x[2]))

    @staticmethod
    def get_random_point(a, b, c, d):
        return [np.random.uniform(a, b), np.random.uniform(c, d)]

    @staticmethod
    def no_weigh(i, k):
        return 1

    @staticmethod
    def linear_weight(i, k):
        return (k + 1 - i) / k

    @staticmethod
    def exponential_weight(i, k):
        return 1 / pow(2.715, i)

    @staticmethod
    def classify_by_k_neighbours(neighbours, weigh_function):
        result = dict()
        k = len(neighbours)
        for index, neighbour in enumerate(neighbours):
            try:
                result[neighbour[1]] += weigh_function(index, k)
            except KeyError:
                result[neighbour[1]] = weigh_function(index, k)
        return max(result, key=lambda key: result[key])

    @staticmethod
    def select_right_k(learning_dots, distance_method, weigh_function):
        result = dict()
        for k in range(1, 25, 1):
            print(k)
            result[k] = ValueBuilder.classify_miss_by_k(learning_dots, distance_method, weigh_function, k)
        print(result)
        return max(result, key=lambda key: result[key])

    @staticmethod
    def classify_miss_by_k(learning_dots, distance_method, weigh_function, k):
        result = 0
        for index, learning_dot in enumerate(learning_dots):
            testing_data = learning_dots[:index] + learning_dots[index + 1:]
            distances = ValueBuilder.distance_to_dots(learning_dot[0], testing_data, distance_method)
            test_class = ValueBuilder.classify_by_k_neighbours(distances[:k], weigh_function)
            if test_class == learning_dot[1]:
                result += 1
            else:
                continue
        return result
