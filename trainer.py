import model

class trainer(object):
    def __init__(self):

        self.grasp =

        super(trainer, self).__init__()

    def forward(self, color_heightmap, depth_heightmap):
        # Pass input data through model
        output_prob, state_feat = self.model(color_heightmap, depth_heightmap)

        return output_prob, state_feat