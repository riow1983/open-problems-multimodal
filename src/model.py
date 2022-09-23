import torch
import torch.nn as nn

# class MultiTaskModel(nn.Module):
#     """
#     Creates a MTL model with the encoder from "arch" and with dropout multiplier ps.
#     """
#     def __init__(self, arch, ps=0.5):
#         super(MultiTaskModel,self).__init__()
#         self.encoder = create_body(arch)        #fastai function that creates an encoder given an architecture
#         self.fc1 = create_head(1024,1,ps=ps)    #fastai function that creates a head
#         self.fc2 = create_head(1024,2,ps=ps)
#         self.fc3 = create_head(1024,5,ps=ps)

#     def forward(self,x):

#         x = self.encoder(x)
#         age = torch.sigmoid(self.fc1(x))
#         gender = self.fc2(x)
#         ethnicity = self.fc3(x)

#         return [age, gender, ethnicity]



class MultipleRegression(nn.Module):
    def __init__(self, args):
        super(MultipleRegression, self).__init__()
        self.num_tasks = args.num_tasks
        self.num_features = args.num_features

        self.layer_1 = nn.Linear(self.num_features, 2000)
        self.layer_2 = nn.Linear(2000, 200)
        self.layer_3 = nn.Linear(200, 100)
        self.layer_4 = nn.Linear(100, 50)
        self.layer_out = nn.Linear(50, 1)
        # self.layer_out = nn.Linear(50, self.num_tasks)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))

        outs = []
        for task in range(self.num_tasks):
            outs.append(self.layer_out(x))

        return outs
        # x = self.layer_out(x)
        # return x