import numpy as np
from manimlib import *


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        square = Square()

        self.play(ShowCreation(square))
        self.wait()
        self.play(ReplacementTransform(square, circle))
        self.wait()


class PlotCurve(Scene):
    def construct(self):
        axes = Axes((-2 * np.pi, 6 * np.pi), (-3, 3), height=6)
        axes.add_coordinate_labels()
        self.play(Write(axes, lag_ratio=0.01, run_time=1))
        sin_graph = axes.get_graph(
            lambda x: 2 * math.sin(x),
            color=BLUE,
        )
        sin_label = axes.get_graph_label(sin_graph, label=r"y=2\sin(x)")
        self.play(
            ShowCreation(sin_graph),
            FadeIn(sin_label, RIGHT),
        )
        self.wait()
