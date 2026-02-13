import ratinabox

from scipy.special import softmax
import copy
import pprint
import numpy as np
from scipy import stats as stats

from ratinabox import utils
from ratinabox.Environment import Environment

"""NEURONS"""
"""Parent Class"""


class Neurons:
    """The Neuron class defines a population of Neurons. All Neurons have firing rates which depend on the state of the Agent. As the Agent moves the firing rate of the cells adjust accordingly.

    All Neuron classes must be initalised with the Agent (to whom these cells belong) since the Agent determines the firingrates through its position and velocity. The Agent class will itself contain the Environment. Both the Agent (position/velocity) and the Environment (geometry, walls, objects etc.) determine the firing rates. Optionally (but likely) an input dictionary 'params' specifying other params will be given.

    This is a generic Parent class. We provide several SubClasses of it. These include:
    • PlaceCells()
    • GridCells()
    • BoundaryVectorCells()
    • ObjectVectorCells()
    • AgentVectorCells()
    • FieldOfViewBVCs()
    • FieldOfViewOVCs()
    • FieldOfViewAVCs()
    • VelocityCells()
    • HeadDirectionCells()
    • SpeedCells()
    • FeedForwardLayer()
    • RandomSpatialNeurons()
    as well as (in  the contribs)
    • ValueNeuron()
    • NeuralNetworkNeurons()

    The unique function in each child classes is get_state(). Whenever Neurons.update() is called Neurons.get_state() is then called to calculate and return the firing rate of the cells at the current moment in time. This is then saved. In order to make your own Neuron subclass you will need to write a class with the following mandatory structure:

    ============================================================================================
    MyNeuronClass(Neurons):

        default_params = {'a_default_param":3.14159} # default params dictionary is defined in the preamble, as a class attribute. Note its values are passed upwards and used in all the parents classes of your class.

        def __init__(self,
                     Agent,
                     params={}): #<-- do not change these


            self.params = copy.deepcopy(__class__.default_params) # to get the default param dictionary of the current class, defined in the preamble, use __class__. Then, make sure to deepcopy it, as only making a shallow copy can have unintended consequences (i.e., any modifications to it would be propagated to ALL instances of this class!).
            self.params.update(params)

            super().__init__(Agent,self.params)

        def get_state(self,
                      evaluate_at='agent',
                      **kwargs) #<-- do not change these

            firingrate = .....
            ###
                Insert here code which calculates the firing rate.
                This may work differently depending on what you set evaluate_at as. For example, evaluate_at == 'agent' should mean that the position or velocity (or whatever determines the firing rate) will by evaluated using the agents current state. You might also like to have an option like evaluate_at == "all" (all positions across an environment are tested simultaneously - plot_rate_map() tries to call this, for example) or evaluate_at == "last" (in a feedforward layer just look at the last firing rate saved in the input layers saves time over recalculating them.). **kwargs allows you to pass position or velocity in manually.

                By default, the Neurons.update() calls Neurons.get_state() rwithout passing any arguments. So write the default behaviour of get_state() to be what you want it to do in the main training loop, .
            ###

            return firingrate

        def any_other_functions_you_might_want(self):...
    ============================================================================================

    As we have written them, Neuron subclasses which have well defined ground truth spatial receptive fields (PlaceCells, GridCells but not VelocityCells etc.) can also be queried for any arbitrary pos/velocity (i.e. not just the Agents current state) by passing these in directly to the function "get_state(evaluate_at='all') or get_state(evaluate_at=None, pos=my_array_of_positons)". This calculation is vectorised and relatively fast, returning an array of firing rates one for each position. It is what is used when you try Neuron.plot_rate_map().

    List of key functions...
        ..that you're likely to use:
            • update()
            • plot_rate_timeseries()
            • plot_rate_map()
        ...that you might not use but could be useful:
            • save_to_history()
            • reset_history()
            • boundary_vector_preference_function()
    """

    default_params = {
        "n": 10,
        "name": "Neurons",
        "color": None,  # just for plotting
        "noise_std": 0,  # 0 means no noise, std of the noise you want to add (Hz)
        "noise_coherence_time": 0.5,
        "min_fr":0.0, #not all cells use max_fr nd min_fr but we define them here in the parent class for those that do 
        "max_fr":1.0,
        "save_history": True,  # whether to save history (set to False if you don't intend to access Neuron.history for data after, for better memory performance)
    }

    def __init__(self, params={}):
        """Initialise Neurons(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.

        Typically you will not actually initialise a Neurons() class, instead you will initialised by one of it's subclasses.
        """

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        utils.update_class_params(self, self.params, get_all_defaults=True)
        utils.check_params(self, params.keys())

        self.firingrate = np.zeros(self.n)
        self.noise = np.zeros(self.n)
        self.history = {}
        self.history["t"] = []
        self.history["firingrate"] = []
        self.history["spikes"] = []

        self._last_history_array_cache_time = None
        self._history_arrays = {} # this dictionary is the same as self.history except the data is in arrays not lists BUT it should only be accessed via its getter-function `self.get_history_arrays()`. This is because the lists are only converted to arrays when they are accessed, not on every step, so as to save time.

        self.colormap = "inferno" # default colormap for plotting ratemaps 

        if ratinabox.verbose is True:
            print(
                f"\nA Neurons() class has been initialised with parameters f{self.params}. Use Neurons.update() to update the firing rate of the Neurons to correspond with the Agent.Firing rates and spikes are saved into the Agent.history dictionary. Plot a timeseries of the rate using Neurons.plot_rate_timeseries(). Plot a rate map of the Neurons using Neurons.plot_rate_map()."
            )

    @classmethod
    def get_all_default_params(cls, verbose=False):
        """Returns a dictionary of all the default parameters of the class, including those inherited from its parents."""
        all_default_params = utils.collect_all_params(cls, dict_name="default_params")
        if verbose:
            pprint.pprint(all_default_params)
        return all_default_params

    def get_state(self, **kwargs):
        raise NotImplementedError("Neurons object needs a get_state() method")

"""Specific subclasses """




class GridCells(Neurons):
    """The GridCells class defines a population of 'n' grid cells with orientations, grid scales and offsets (these can be set randomly or non-randomly). Grids are modelled as the rectified or shifted sum of three cosine waves at 60 degrees to each other.

    To initialise grid cells you specify their (i) params['gridscale'], (ii) params['orientation'] and (iii) params['phase_offset']. These can be handed in as lists/arrays (in which case they are set to these exact values, one per cell) or tuples where the values inside the tuples define the parameters of a distribution (the string defined by params['<param>_distribution']) from which the parameters are sampled. An up-to-date list of avaiable distributions and their parameters in utils.distribution_sampler(), currently avaiable distributions are:
    - uniform ------------------------------- (low, high) or just a single param p which gives (0.5*p, 1.5*p)
    - rayleigh ------------------------------ (scale)
    - normal -------------------------------- (loc, scale)
    - logarithmic --------------------------- (low, high)
    - delta --------------------------------- (the_single_value)
    - modules ------------------------------- (module1_val, module2_val, module3_val, ...)
    - truncnorm ----------------------------- (low, high, loc, scale)
    For example to get three modules of gridcells I could set params = {'gridscale_distribution':'modules', 'gridscale':(0.5, 1, 1.5)} which would give me three modules of grid cells with grid scales 0.5, 1 and 1.5. 

    List of functions:
        • get_state()
        • set_phase_offsets()

        }
    """

    default_params = {
        "n": 100,
        "name": "GridCells",
        "gridscale_distribution": "modules",
        "gridscale": (0.3, 0.5, 0.8),
        "orientation_distribution": "modules",
        "orientation": (0, 0.1, 0.2), #radians 
        "phase_offset_distribution": "uniform",
        "phase_offset": (0, 2 * np.pi), #degrees
        "width_ratio":4/(3*np.sqrt(3)), # (relevant only for "rectified_cosines") ratio of field_width to interfield_distance.
        "min_fr": 0,
        "max_fr": 1,
    }

    def __init__(self, params={}):
        """Initialise GridCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
        """

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        # Initialise the gridscales
        if type(self.params["gridscale"]) in (
            list,
            np.ndarray,
        ):  # assumes you are manually passing gridscales, one for each neuron
            self.gridscales = np.array(self.params["gridscale"])
            self.params["n"] = len(self.gridscales)
        elif type(self.params["gridscale"]) in (
            float,
            tuple,
            int,
        ):  # assumes you are passing distribution parameters
            self.gridscales = utils.distribution_sampler(
                distribution_name=self.params["gridscale_distribution"],
                distribution_parameters=self.params["gridscale"],
                shape=(self.params["n"],),
            )

        # Initialise Neurons parent class
        super().__init__(self.params)

        self.phase_offsets = utils.distribution_sampler(
            distribution_name=self.params["phase_offset_distribution"],
            distribution_parameters=self.params["phase_offset"],
            shape=(self.params["n"], 2),
        )

        # Initialise orientations for each grid cell, only relevant in 2D
        if type(self.params["orientation"]) in (
            list,
            np.ndarray,
        ):
            self.orientations = np.array(self.params["orientation"])
            assert (
                len(self.orientations) == self.params["n"]
            ), "number of orientations supplied incompatible with number of neurons"
        else:
            self.orientations = utils.distribution_sampler(
                distribution_name=self.params["orientation_distribution"],
                distribution_parameters=self.params["orientation"],
                shape=(self.params["n"],),
            )

        w = []
        for i in range(self.n):
            w1 = np.array([1, 0])
            w1 = utils.rotate(w1, self.orientations[i])
            w2 = utils.rotate(w1, np.pi / 3)
            w3 = utils.rotate(w1, 2 * np.pi / 3)
            w.append(np.array([w1, w2, w3]))
        self.w = np.array(w)
        
        assert self.width_ratio > 0 and self.width_ratio <= 1, "width_ratio must be between 0 and 1"

        if ratinabox.verbose is True:
            print(
                "GridCells successfully initialised. You can also manually set their gridscale (GridCells.gridscales), offsets (GridCells.phase_offsets) and orientations (GridCells.w1, GridCells.w2,GridCells.w3 give the cosine vectors)"
            )

    def get_state(self, pos, **kwargs):
        """Returns the firing rate of the grid cells.
        By default position is taken from the Agent and used to calculate firing rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or you can use all the positions in the environment (evaluate_at="all").

        Returns:
            firingrates: an array of firing rates
        """
        pos = np.array(pos)
        original_shape = list(pos.shape[:-1])
        pos = pos.reshape(-1, pos.shape[-1])

        # grid cells are modelled as the thresholded sum of three cosines all at 60 degree offsets
        # vectors to grids cells "centred" at their (random) phase offsets
        origin = self.gridscales.reshape(-1, 1) * self.phase_offsets / (2 * np.pi)
        vecs = utils.get_vectors_between(origin, pos)  # shape = (N_cells,N_pos,2)
        w1 = np.tile(np.expand_dims(self.w[:, 0, :], axis=1), reps=(1, pos.shape[0], 1))
        w2 = np.tile(np.expand_dims(self.w[:, 1, :], axis=1), reps=(1, pos.shape[0], 1))
        w3 = np.tile(np.expand_dims(self.w[:, 2, :], axis=1), reps=(1, pos.shape[0], 1))
        gridscales = np.tile(
            np.expand_dims(self.gridscales, axis=1), reps=(1, pos.shape[0])
        )
        phi_1 = ((2 * np.pi) / gridscales) * (vecs * w1).sum(axis=-1)
        phi_2 = ((2 * np.pi) / gridscales) * (vecs * w2).sum(axis=-1)
        phi_3 = ((2 * np.pi) / gridscales) * (vecs * w3).sum(axis=-1)

        firingrate = (1 / 3) * ((np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)))
        
        #calculate the firing rate at the width fraction then shift, scale and rectify at the level
        a, b, c = np.array([1,0])@np.array([1,0]), np.array([np.cos(np.pi/3),np.sin(np.pi/3)])@np.array([1,0]), np.array([np.cos(np.pi/3),-np.sin(np.pi/3)])@np.array([1,0])
        firing_rate_at_full_width = (1 / 3) * (np.cos(np.pi*self.width_ratio*a) + 
                                        np.cos(np.pi*self.width_ratio*b) + 
                                        np.cos(np.pi*self.width_ratio*c))
        firing_rate_at_full_width = (1 / 3) * (2*np.cos(np.sqrt(3)*np.pi*self.width_ratio/2) + 1)
        firingrate -= firing_rate_at_full_width
        firingrate /= (1 - firing_rate_at_full_width)
        firingrate[firingrate < 0] = 0

        # firingrate = (
        #     firingrate * (self.max_fr - self.min_fr) + self.min_fr
        # )  # scales from being between [0,1] to [min_fr, max_fr]

        if self.params['softmax']:
            firingrate = softmax(firingrate, axis=0)

        firingrate = np.moveaxis(firingrate, 1, 0)

        firingrate = firingrate.reshape(original_shape+[self.n])

        return firingrate

    def set_phase_offsets_on_grid(self):
        """Set non-random phase_offsets. Most offsets (n_on_grid, the largest square numer before self.n) will tile a grid of 0 to 2pi in x and 0 to 2pi in y, while the remainings (cell number: n - n_on_grid) are random."""
        n_x = int(np.sqrt(self.n))
        n_y = self.n // n_x
        n_remaining = self.n - n_x * n_y

        dx = 2 * np.pi / n_x
        dy = 2 * np.pi / n_y

        grid = np.mgrid[
            (0 + dx / 2) : (2 * np.pi - dx / 2) : (n_x * 1j),
            (0 + dy / 2) : (2 * np.pi - dy / 2) : (n_y * 1j),
        ]
        grid = grid.reshape(2, -1).T
        remaining = np.random.uniform(0, 2 * np.pi, size=(n_remaining, 2))

        all_offsets = np.vstack([grid, remaining])

        return all_offsets
