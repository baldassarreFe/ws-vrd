from ignite.engine import Events
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger


class TensorboardLogHandler(BaseHandler):
    def __init__(self, global_step_fn):
        self.global_step_fn = global_step_fn

    def __call__(self, engine, logger: TensorboardLogger, event_name: Events):
        global_step = self.global_step_fn()

        for name, value in engine.state.metrics.items():
            logger.writer.add_scalar(f'{engine.name}/{name}', value, global_step)

        for name, fig in engine.state.figures.items():
            logger.writer.add_figure(f'{engine.name}/{name}', fig, global_step, close=True)

        for name, img in engine.state.images.items():
            logger.writer.add_image(f'{engine.name}/{name}', img, global_step)

        for name, img in engine.state.text.items():
            logger.writer.add_text(f'{engine.name}/{name}', img, global_step)
