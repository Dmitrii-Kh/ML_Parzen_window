import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Slider, RadioButtons, Button
import numpy as np

from parzen_fixed.ValueBuilder import ValueBuilder
from parzen_fixed.GraphicBuilder import GraphicBuilder

if __name__ == "__main__":
    graphic_builder = GraphicBuilder()
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Classification')
    plt.title("Classification")

    fig.set_size_inches(graphic_builder.WINDOW_WIDTH, graphic_builder.WINDOW_HEIGHT)

    vertical_lines = dict()
    horizontal_lines = dict()

    hor_res, ver_res = graphic_builder.box_hor_ver()

    for index, vertical_line in enumerate(graphic_builder.ab_segment):
        vertical_lines[index] = ax.axvline(vertical_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)

    for index, horizontal_line in enumerate(graphic_builder.cd_segment):
        horizontal_lines[index] = ax.axhline(horizontal_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)

    # hor_span = ax.axhspan(ymin=graphic_builder.cd_segment[hor_res[0]], ymax=graphic_builder.cd_segment[hor_res[1]],
    #                       color="grey",
    #                       alpha=0.3)
    # ver_span = ax.axvspan(xmin=graphic_builder.ab_segment[ver_res[0]], xmax=graphic_builder.ab_segment[ver_res[1]],
    #                       color="grey",
    #                       alpha=0.3)

    rect_width = graphic_builder.ab_segment[ver_res[1]] - graphic_builder.ab_segment[ver_res[0]]
    rect_height = graphic_builder.cd_segment[hor_res[1]] - graphic_builder.cd_segment[hor_res[0]]

    # Create a Rectangle patch
    rect = patches.Rectangle((graphic_builder.ab_segment[ver_res[0]], graphic_builder.cd_segment[hor_res[0]]),
                             rect_width, rect_height, linewidth=1.6, edgecolor='g', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    dots_graphic = ax.scatter(graphic_builder.learning_dots_horizontal, graphic_builder.learning_dots_vertical,
                              s=graphic_builder.DOT_LEARNING_SIZE, color=graphic_builder.DOT_LEARNING_COLOR)

    learning_dot = ax.scatter(graphic_builder.random_point[0], graphic_builder.random_point[1],
                              s=graphic_builder.DOT_TESTING_SIZE, color=graphic_builder.DOT_TESTING_COLOR)

    etalons = ax.scatter(graphic_builder.etalons_horizontal, graphic_builder.etalons_vertical,
                         s=graphic_builder.ETALONS_SIZE, color=graphic_builder.ETALONS_COLOR)

    draw_circle = patches.Circle((graphic_builder.random_point[0], graphic_builder.random_point[1]), graphic_builder.h,
                                 fill=False)
    ax.add_patch(draw_circle)

    fig.tight_layout()

    # ax.legend(
    #     [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_LEARNING_COLOR),
    #      Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=graphic_builder.DOT_TESTING_COLOR)],
    #     ("Learning points", "Testing point"),
    #     numpoints=1)

    dots_axes = plt.axes([0.25, 0.03, 0.40, 0.02], facecolor='white')
    dots_slider = Slider(dots_axes, '# points', graphic_builder.MIN_LEARNING_DOTS,
                         graphic_builder.MAX_LEARNING_DOTS,
                         valinit=graphic_builder.l, valstep=10, color='blue')

    vert_axes = plt.axes([0.25, 0.09, 0.40, 0.02], facecolor='white')
    vert_slider = Slider(vert_axes, 'cols', graphic_builder.MIN_CLASSES,
                         graphic_builder.MAX_CLASSES,
                         valinit=graphic_builder.n, valstep=1, color='blue')

    hor_axes = plt.axes([0.25, 0.06, 0.40, 0.02], facecolor='white')
    hor_slider = Slider(hor_axes, 'rows', graphic_builder.MIN_CLASSES,
                        graphic_builder.MAX_CLASSES,
                        valinit=graphic_builder.m, valstep=1, color='blue')

    h_axes = plt.axes([0.25, 0.12, 0.40, 0.02], facecolor='white')
    h_slider = Slider(h_axes, 'h', graphic_builder.MIN_H, graphic_builder.MAX_H, valinit=graphic_builder.h,
                      valstep=0.05, color='blue')

    etalons_axes = plt.axes([0.25, 0.15, 0.40, 0.02], facecolor='white')
    etalons_slider = Slider(etalons_axes, 'Etalons', 0.01, 1, valinit=graphic_builder.etalonsPercentage,
                      valstep=0.01, color='blue')

    rax = plt.axes([0.80, 0.03, 0.15, 0.12])
    radio = RadioButtons(rax, ('Boolean', 'Linear', 'Quadratic', 'Gauss'))

    # rax_distance = plt.axes([0.81, 0.03, 0.15, 0.12])
    # radio_distance = RadioButtons(rax_distance, ('Евклідова', 'Манхетенна'))

    current_miss = ValueBuilder.classify_miss_by_h(graphic_builder.classified_dots, ValueBuilder.euclidian_distance,
                                                   ValueBuilder.linear_kernel, graphic_builder.h)

    loss_text = fig.text(0.05, 0.95, s=f"Total hit: {current_miss}/{graphic_builder.l}", fontsize=14)

    kernel_dict = {'Boolean': ValueBuilder.boolean_kernel, 'Linear': ValueBuilder.linear_kernel,
                   'Quadratic': ValueBuilder.quadratic_kernel, 'Gauss': ValueBuilder.gauss_kernel}

    distance_dict = {'Евклідова': ValueBuilder.euclidian_distance, 'Манхетенна': ValueBuilder.manhattan_distance}

    button_axes = plt.axes([0.05, 0.08, 0.05, 0.05])
    button = Button(button_axes, 'Learn')

    random_axes = plt.axes([0.05, 0.14, 0.05, 0.05])
    random_dot = Button(random_axes, 'New point')

    calculate_axes = plt.axes([0.05, 0.02, 0.05, 0.05])
    calculate_button = Button(calculate_axes, 'Calculate\nefficiency')


    def calculate_hit(event):
        current_miss = ValueBuilder.classify_miss_by_h(graphic_builder.classified_dots,
                                                       graphic_builder.DISTANCE_ALGORITHM,
                                                       graphic_builder.KERNEL, graphic_builder.h)
        loss_text.set_text(f"Total hit: {current_miss}/{graphic_builder.l}")


    def update_spans_classify():
        graphic_builder.classify_dot()

        # graphic_builder.etalons_vertical = []
        # graphic_builder.etalons_horizontal = []
        # etalon_dots = list(
        #     sorted(np.column_stack((graphic_builder.etalons_horizontal, graphic_builder.etalons_vertical)),
        #            key=lambda x: (x[0], x[1])))
        # etalons.set_offsets(etalon_dots)

        hor_res, ver_res = graphic_builder.box_hor_ver()
        [p.remove() for p in reversed(ax.patches)]

        rect_width = graphic_builder.ab_segment[ver_res[1]] - graphic_builder.ab_segment[ver_res[0]]
        rect_height = graphic_builder.cd_segment[hor_res[1]] - graphic_builder.cd_segment[hor_res[0]]

        # Create a Rectangle patch
        rect = patches.Rectangle((graphic_builder.ab_segment[ver_res[0]], graphic_builder.cd_segment[hor_res[0]]),
                                 rect_width, rect_height, linewidth=1.6, edgecolor='g', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        draw_circle = patches.Circle((graphic_builder.random_point[0], graphic_builder.random_point[1]),
                                     graphic_builder.h, fill=False)
        ax.add_patch(draw_circle)

        # global hor_span, ver_span
        # try:
        #     hor_span.remove()
        #     ver_span.remove()
        # except ValueError:
        #     pass
        # hor_span = ax.axhspan(graphic_builder.cd_segment[hor_res[0]], graphic_builder.cd_segment[hor_res[1]],
        #                       color="grey",
        #                       alpha=0.3)
        # ver_span = ax.axvspan(graphic_builder.ab_segment[ver_res[0]], graphic_builder.ab_segment[ver_res[1]],
        #                       color="grey",
        #                       alpha=0.3)


    def update_distance(label):
        fig.canvas.draw_idle()
        graphic_builder.DISTANCE_ALGORITHM = distance_dict[label]
        update_spans_classify()


    def update_weight(label):
        fig.canvas.draw_idle()
        graphic_builder.KERNEL = kernel_dict[label]
        update_spans_classify()


    def update_h(val):
        fig.canvas.draw_idle()
        graphic_builder.h = float(val)
        update_spans_classify()

    def update_etalons(val):
        fig.canvas.draw_idle()
        graphic_builder.etalons_vertical = []
        graphic_builder.etalons_horizontal = []
        graphic_builder.etalonsPercentage = float(val)
        update_spans_classify()
        etalon_dots = list(
            sorted(np.column_stack((graphic_builder.etalons_horizontal, graphic_builder.etalons_vertical)),
                   key=lambda x: (x[0], x[1])))
        etalons.set_offsets(etalon_dots)


    def update_n(val):
        fig.canvas.draw_idle()
        graphic_builder.etalons_vertical = []
        graphic_builder.etalons_horizontal = []
        graphic_builder.n = int(val)
        graphic_builder.generate_ab_and_classify()
        global vertical_lines
        for vertical_line in list(vertical_lines):
            vertical_lines[vertical_line].remove()
            vertical_lines.pop(vertical_line, None)
        for index, vertical_line in enumerate(graphic_builder.ab_segment):
            vertical_lines[index] = ax.axvline(vertical_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)
        update_spans_classify()
        etalon_dots = list(
            sorted(np.column_stack((graphic_builder.etalons_horizontal, graphic_builder.etalons_vertical)),
                   key=lambda x: (x[0], x[1])))
        etalons.set_offsets(etalon_dots)


    def update_m(val):
        fig.canvas.draw_idle()
        graphic_builder.etalons_vertical = []
        graphic_builder.etalons_horizontal = []
        graphic_builder.m = int(val)
        graphic_builder.generate_cd_and_classify()
        global horizontal_lines
        for hor_line in list(horizontal_lines):
            horizontal_lines[hor_line].remove()
            horizontal_lines.pop(hor_line, None)
        for index, horizontal_line in enumerate(graphic_builder.cd_segment):
            horizontal_lines[index] = ax.axhline(horizontal_line, linestyle="-.", linewidth=graphic_builder.LINE_WIDTH)
        update_spans_classify()
        etalon_dots = list(
            sorted(np.column_stack((graphic_builder.etalons_horizontal, graphic_builder.etalons_vertical)),
                   key=lambda x: (x[0], x[1])))
        etalons.set_offsets(etalon_dots)


    def update_dots(val):
        fig.canvas.draw_idle()
        graphic_builder.etalons_vertical = []
        graphic_builder.etalons_horizontal = []
        graphic_builder.l = int(val)
        graphic_builder.generate_dots_and_clasify()
        graphic_builder.classify_dot()
        update_spans_classify()
        dots_graphic.set_offsets(graphic_builder.learning_dots)
        etalon_dots = list(
            sorted(np.column_stack((graphic_builder.etalons_horizontal, graphic_builder.etalons_vertical)),
                   key=lambda x: (x[0], x[1])))
        etalons.set_offsets(etalon_dots)


    def compute_h(event):
        fig.canvas.draw_idle()
        graphic_builder.h = ValueBuilder.select_right_h(graphic_builder.classified_dots,
                                                        graphic_builder.DISTANCE_ALGORITHM,
                                                        graphic_builder.KERNEL)
        h_slider.set_val(graphic_builder.h)
        calculate_hit(event)


    def generate_new(event):
        fig.canvas.draw_idle()
        graphic_builder.generate_and_clasify_dot()
        hor_res, ver_res = graphic_builder.box_hor_ver()

        [p.remove() for p in reversed(ax.patches)]
        rect_width = graphic_builder.ab_segment[ver_res[1]] - graphic_builder.ab_segment[ver_res[0]]
        rect_height = graphic_builder.cd_segment[hor_res[1]] - graphic_builder.cd_segment[hor_res[0]]

        # Create a Rectangle patch
        rect = patches.Rectangle((graphic_builder.ab_segment[ver_res[0]], graphic_builder.cd_segment[hor_res[0]]),
                                 rect_width, rect_height, linewidth=1.6, edgecolor='g', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        # global hor_span, ver_span
        # try:
        #     hor_span.remove()
        #     ver_span.remove()
        # except ValueError:
        #     pass
        # hor_span = ax.axhspan(graphic_builder.cd_segment[hor_res[0]], graphic_builder.cd_segment[hor_res[1]],
        #                       color="grey",
        #                       alpha=0.3)
        # ver_span = ax.axvspan(graphic_builder.ab_segment[ver_res[0]], graphic_builder.ab_segment[ver_res[1]],
        #                       color="grey",
        #                       alpha=0.3)
        learning_dot.set_offsets([graphic_builder.random_point[0], graphic_builder.random_point[1]])
        draw_circle = patches.Circle((graphic_builder.random_point[0], graphic_builder.random_point[1]),
                                     graphic_builder.h, fill=False)
        ax.add_patch(draw_circle)


    radio.on_clicked(update_weight)
    #   radio_distance.on_clicked(update_distance)
    h_slider.on_changed(update_h)
    etalons_slider.on_changed(update_etalons)
    vert_slider.on_changed(update_n)
    hor_slider.on_changed(update_m)
    dots_slider.on_changed(update_dots)
    button.on_clicked(compute_h)
    calculate_button.on_clicked(calculate_hit)
    random_dot.on_clicked(generate_new)
    plt.subplots_adjust(bottom=0.235, right=0.98)
    plt.show()
