As the code above, we know what each component is responsible for and how the components are connected together to make the base implementation work. The next step is to pick one approach to improve our code. We are about to improve the performance using Dueling DQN. 

We can explain the reason and inspiration of Dueling DQN in short: the value of ***Q(s,a)*** represet the reward value of action ***a*** in state ***s***, however, in some state, no matter what action you take, it always gets high reward, and oppositely in some state, no matter what action you take, it always gets low reward. We can use this feature to improve the neural network we use to approximate the value. 

So we can divide the neural network to two piece, one use state ***s*** as input, called state value function, and the other one use state ***s*** and action ***a*** as input, called state-dependent action advantage function:

<img src=".\pic1.png" alt="pic1" style="zoom:67%;" />

The equation is unidentifiable in the sense that given Q we cannot recover V and A uniquely. So we can alter the module to be like:

<img src=".\pic2.png" style="zoom:67%;" />

After this, we can get our Q neural network like this(shown in table like a Q-learning table):

<img src=".\pic3.jpg" style="zoom:50%;" />

As the picture above, $\begin{equation*}\sum_{}^{}{A(s,\_)}=0 \end{equation*}$ and we use the sum of S and A to get Q value. And in the code implementation, we just need to adjust several rows in the ***utils_model.py*** in class DQN:

```python
class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(512, action_dim)

        
        self.__s1 = nn.Linear(64*7*7, 512)
        self.__s2 = nn.Linear(512, 1)

        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))

        sa = F.relu(self.__fc1(x.view(x.size(0), -1)))
        sa = self.__fc2(sa)

        s = F.relu(self.__fc1(x.view(x.size(0), -1)))
        s = self.__fc2(s)

        d = torch.mean(sa, axis=1, keepdim=True)

        return s+sa-d 

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
```





