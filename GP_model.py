import torch
import gpytorch

def normalize_for_train(inputs, target):
    inputs = torch.stack([torch.tensor(inputs[i]) for i in range(len(inputs))], dim=1).float()
    target = torch.tensor(target).float()
    means_in = inputs.mean(0)
    stds_in = inputs.std(0) + 1e-6
    inputs = (inputs - means_in) / stds_in
    mean_out = target.mean()
    std_out = target.std() + 1e-6
    target = (target - mean_out) / std_out
    scaling = {'means_in': means_in, 'stds_in': stds_in, 'mean_out': mean_out, 'std_out': std_out}
    return inputs, target, scaling

def scale_for_predict(inputs, scaling):
    inputs = torch.stack([torch.tensor(inputs[i]) for i in range(len(inputs))], dim=1).float()
    inputs = (inputs - scaling['means_in']) / scaling['stds_in']
    return inputs

def rescale_output(output, scaling):
    output = output * scaling['std_out'] + scaling['mean_out']
    return output

# We will use the simplest form of GP model, exact inference
class ExactGPModel2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel2D, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp_model(train_x, train_y, n_train = 50):
    ''' Fit a GP model to scaled input and output data'''
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel2D(train_x, train_y, likelihood)
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(n_train):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, n_train, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    model.cpu()
    likelihood.cpu()
    return model
    

def predict_gp_model(model, predict_x):
    ''' Predict using the fitted GP model'''
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(predict_x)
        mean = observed_pred.mean
        std = observed_pred.stddev
    return mean, std
