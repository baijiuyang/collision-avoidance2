'''
This module contains all the trained model parameters
'''

# Define approach models for fitting avoidance models
approaches = {}
approaches['Bai_movObst1'] = {'fajen_approach': {}, 'fajen_approach2': {}, 'acceleration_approach': {}, 'jerk_approach': {}}
approaches['Bai_movObst1b'] = {'fajen_approach': {}, 'fajen_approach2': {}, 'acceleration_approach': {}, 'jerk_approach': {}}

approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'] = {}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][-1] = {'name': 'fajen_approach', 'b1': 4.296329212568901, 'k1': 3.876411184381646, 'c1': 2.373202635931417, 'c2': 1.0940365636783889, 'k2': 1.0565235274635225}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][0] = {'name': 'fajen_approach', 'b1': 5.171374631693293, 'k1': 3.7748704888285456, 'c1': 0.2524727891811863, 'c2': 0.9670404267366893, 'k2': 0.9778603110291152}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][1] = {'name': 'fajen_approach', 'b1': 4.687275769750559, 'k1': 1.6513680556736985, 'c1': 0.3040389605597095, 'c2': 2.6818082947581727, 'k2': 1.3711088659320352}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][2] = {'name': 'fajen_approach', 'b1': 3.2242608458522133, 'k1': 6.598949320560229, 'c1': 0.5992242093062996, 'c2': 0.5130353096053771, 'k2': 0.9606261544156356}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][3] = {'name': 'fajen_approach', 'b1': 2.1553858457123383, 'k1': 2.4250753925187736, 'c1': 0.5379235884706337, 'c2': 0.9395101950577249, 'k2': 1.2778585399685836}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][4] = {'name': 'fajen_approach', 'b1': 3.013949086307595, 'k1': 1.9714483648445507, 'c1': 0.47859753218527606, 'c2': 1.5331239229201745, 'k2': 1.0840935091122041}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][6] = {'name': 'fajen_approach', 'b1': 4.420933310055661, 'k1': 3.5706426083503446, 'c1': 0.3146817674488948, 'c2': 0.7220059641040694, 'k2': 1.1120585827002478}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][7] = {'name': 'fajen_approach', 'b1': 3.853879812194764, 'k1': 3.223946745381489, 'c1': 0.3853634365918022, 'c2': 1.4075153762917059, 'k2': 1.6695892746375836}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][9] = {'name': 'fajen_approach', 'b1': 8.412773060911043, 'k1': 7.576581326417636, 'c1': 0.32595839644972346, 'c2': 0.713326049149128, 'k2': 0.822573324027422}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][10] = {'name': 'fajen_approach', 'b1': 4.717438327725703, 'k1': 2.6515951111242804, 'c1': 0.4538067831085721, 'c2': 1.1165981281203992, 'k2': 0.8160458132648651}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][11] = {'name': 'fajen_approach', 'b1': 2.812970031319312, 'k1': 2.384120199630047, 'c1': 3.0918480881358144, 'c2': 1.3587018979322045, 'k2': 0.7928858954051313}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][12] = {'name': 'fajen_approach', 'b1': 5.472415932291602, 'k1': 3.6570987973910443, 'c1': 0.2448513181611536, 'c2': 0.8575130568802705, 'k2': 1.0810204178683263}
approaches['Bai_movObst1b']['fajen_approach']['differential_evolution'][13] = {'name': 'fajen_approach', 'b1': 4.183141558927925, 'k1': 6.26502602722788, 'c1': 0.4605941086625824, 'c2': 0.5478517466755397, 'k2': 0.9681391241515049}

approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'] = {}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][-1] = {'name': 'fajen_approach2', 'b1': 3.7380520763369196, 'k1': 2.87947916970264, 'c1': 0.2206162881805807, 'c2': 0.960123052819782, 'b2': 2.359847074014972, 'k2': 2.570997808604569}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][0] = {'name': 'fajen_approach2', 'b1': 4.685325039598137, 'k1': 6.9855179157418155, 'c1': 0.5320768638356439, 'c2': 0.5268734750702514, 'b2': 4.178247872327981, 'k2': 4.094538191141033}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][1] = {'name': 'fajen_approach2', 'b1': 4.877686344068194, 'k1': 1.886022196454624, 'c1': 0.2631880832575354, 'c2': 2.3470280774261414, 'b2': 2.741205811548106, 'k2': 3.7409117799177496}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][2] = {'name': 'fajen_approach2', 'b1': 2.7188629441551027, 'k1': 3.368744419890986, 'c1': 0.2383054872129715, 'c2': 0.6720200216353054, 'b2': 3.29234668702906, 'k2': 3.2958464957519973}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][3] = {'name': 'fajen_approach2', 'b1': 2.3009874286387535, 'k1': 1.7697759047952653, 'c1': 4.075942164242822, 'c2': 1.498451942346544, 'b2': 2.7737417670028695, 'k2': 3.5764157256875557}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][4] = {'name': 'fajen_approach2', 'b1': 2.894962564521153, 'k1': 1.2822457997984065, 'c1': 0.24855407698568194, 'c2': 2.151926726710701, 'b2': 2.592221307241384, 'k2': 2.9209177936258683}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][6] = {'name': 'fajen_approach2', 'b1': 5.056242688735129, 'k1': 2.658255212225606, 'c1': 0.23114083410108702, 'c2': 1.0949314221719848, 'b2': 2.304749263658365, 'k2': 2.5332510646561786}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][7] = {'name': 'fajen_approach2', 'b1': 3.868166988451832, 'k1': 1.3488392821205601, 'c1': 0.21291899282713, 'c2': 3.3654229635731836, 'b2': 1.7641173702180104, 'k2': 2.6966150116658043}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][9] = {'name': 'fajen_approach2', 'b1': 8.006003239906777, 'k1': 1.9077017222135435, 'c1': 0.24254561076127432, 'c2': 3.1096529591782973, 'b2': 3.2273105804422095, 'k2': 2.8563388977254025}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][10] = {'name': 'fajen_approach2', 'b1': 4.294586773127741, 'k1': 10.940580271301643, 'c1': 0.7249070798631742, 'c2': 0.230122685199928, 'b2': 3.129184115912155, 'k2': 2.662932210845456}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][11] = {'name': 'fajen_approach2', 'b1': 2.6686147220499743, 'k1': 5.738830402782805, 'c1': 0.778277559046697, 'c2': 0.4992475894022052, 'b2': 3.3906476034125292, 'k2': 2.8279395254386674}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][12] = {'name': 'fajen_approach2', 'b1': 7.7855258192068435, 'k1': 4.068891048830318, 'c1': 0.3066495719696504, 'c2': 1.2196321300214885, 'b2': 2.70757462588659, 'k2': 3.0054711131627405}
approaches['Bai_movObst1b']['fajen_approach2']['differential_evolution'][13] = {'name': 'fajen_approach2', 'b1': 4.075907968599436, 'k1': 9.243017312440472, 'c1': 0.5831183555340775, 'c2': 0.36162337833154756, 'b2': 2.329812666404041, 'k2': 2.347732705113913}

approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'] = {}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][-1] = {'name': 'acceleration_approach', 'k': 1.0077701621761348}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][0] = {'name': 'acceleration_approach', 'k': 0.9308004941780045}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][1] = {'name': 'acceleration_approach', 'k': 1.2504892636683487}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][2] = {'name': 'acceleration_approach', 'k': 1.0011736637034616}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][3] = {'name': 'acceleration_approach', 'k': 1.237658962333328}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][4] = {'name': 'acceleration_approach', 'k': 1.0683654265297895}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][6] = {'name': 'acceleration_approach', 'k': 0.9468803263681959}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][7] = {'name': 'acceleration_approach', 'k': 1.5253691915476106}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][9] = {'name': 'acceleration_approach', 'k': 0.8028245525525753}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][10] = {'name': 'acceleration_approach', 'k': 0.7751821455389301}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][11] = {'name': 'acceleration_approach', 'k': 0.8327326015091421}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][12] = {'name': 'acceleration_approach', 'k': 0.9470182154831401}
approaches['Bai_movObst1b']['acceleration_approach']['differential_evolution'][13] = {'name': 'acceleration_approach', 'k': 0.9674342865463593}

approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][-1] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][0 ] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][1] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][2] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][3] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][4] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][6] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][7] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][9] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][10] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][11] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][12] = {}
approaches['Bai_movObst1b']['jerk_approach']['differential_evolution'][13] = {}

approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'] = {}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][-1] = {'name': 'fajen_approach', 'ps': None, 'b1': 3.25, 'k1': 7.5, 'c1': 0.4, 'c2': 0.4, 'k2': 1.4}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][0] = {'name': 'fajen_approach', 'b1': 1.3118996599538175, 'k1': 20.05817176956081, 'c1': 2.533333934342072, 'c2': 0.07555087088494414, 'k2': 0.7413974342994933, 'ps': 1.282479471344188}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][1] = {'name': 'fajen_approach', 'b1': 0.8081777207885088, 'k1': 32.80043575704669, 'c1': 8.416453555226326, 'c2': 0.03974207207189774, 'k2': 0.8136444335415792, 'ps': 1.329331601172892}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][2] = {'name': 'fajen_approach', 'b1': 0.8802412636720522, 'k1': 18.997479919723922, 'c1': 2.756441843773068, 'c2': 0.0760314647580457, 'k2': 0.9161786337522445, 'ps': 1.2233195327965638}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][3] = {'name': 'fajen_approach', 'b1': 1.2314286830727728, 'k1': 24.035668545742503, 'c1': 9.856130794680212, 'c2': 0.02862746960028267, 'k2': 0.2117503203346745, 'ps': 0.9872111178298977}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][4] = {'name': 'fajen_approach', 'b1': 0.9183592702325633, 'k1': 12.204919745808612, 'c1': 3.43625676096417, 'c2': 0.11664808708772718, 'k2': 0.8674515064029045, 'ps': 1.4286737076029379}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][5] = {'name': 'fajen_approach', 'b1': 1.192778944853645, 'k1': 5.267999799725562, 'c1': 5.096005901694298, 'c2': 0.2242800263696644, 'k2': 0.1765803907428338, 'ps': 1.0751463986115937}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][6] = {'name': 'fajen_approach', 'b1': 1.0735811813874558, 'k1': 13.140505400771387, 'c1': 9.98596467077732, 'c2': 0.11649019178319157, 'k2': 0.586444306946999, 'ps': 1.31109726329129}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][7] = {'name': 'fajen_approach', 'b1': 1.7471982824310022, 'k1': 5.117879113857131, 'c1': 5.741959801035067, 'c2': 0.21995468988652767, 'k2': 0.5773546487545462, 'ps': 1.156007882055671}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][8] = {'name': 'fajen_approach', 'b1': 1.1656785334379702, 'k1': 64.75154006900311, 'c1': 2.39434327185181, 'c2': 0.023455554827526214, 'k2': 0.3991609059701808, 'ps': 1.1563239895464452}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][9] = {'name': 'fajen_approach', 'b1': 1.1927778066346024, 'k1': 0.20667325443360973, 'c1': 2.2570241856267543, 'c2': 5.4331038962018186, 'k2': 0.37382088279617776, 'ps': 1.1967108147865295}
approaches['Bai_movObst1b']['fajen_approach']['dual_annealing'][10] = {'name': 'fajen_approach', 'b1': 1.1996934237333374, 'k1': 11.292256811441096, 'c1': 1.7756010015861, 'c2': 0.17150790956230327, 'k2': 0.4351205520904613, 'ps': 1.056705515676128}

approaches['Bai_movObst1']['fajen_approach']['differential_evolution'] = {}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][-1] = {'name': 'fajen_approach', 'ps': None, 'b1': 3.25, 'k1': 7.5, 'c1': 0.4, 'c2': 0.4, 'k2': 1.4}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][0] = {'name': 'fajen_approach', 'b1': 1.322399694018276, 'k1': 14.876666639462048, 'c1': 5.4374557807911295, 'c2': 0.10260590846412238, 'k2': 0.7564633997176441, 'ps': 1.282479471344188}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][1] = {'name': 'fajen_approach', 'b1': 0.8179416797170797, 'k1': 16.40077656118177, 'c1': 6.417008356831993, 'c2': 0.07993938648972572, 'k2': 0.8075364662966444, 'ps': 1.329331601172892}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][2] = {'name': 'fajen_approach', 'b1': 0.8909408398978989, 'k1': 4.641459981824929, 'c1': 5.516867857632004, 'c2': 0.31100646722160424, 'k2': 0.9262495070122934, 'ps': 1.2233195327965638}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][3] = {'name': 'fajen_approach', 'b1': 1.2318155085696687, 'k1': 0.44596523668254273, 'c1': 8.660447058122513, 'c2': 1.5467823698282144, 'k2': 0.2064201810751935, 'ps': 0.9872111178298977}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][4] = {'name': 'fajen_approach', 'b1': 0.9187400116180737, 'k1': 12.354994295866103, 'c1': 5.138764981999261, 'c2': 0.1150464883117947, 'k2': 0.8664462743167836, 'ps': 1.4286737076029379}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][5] = {'name': 'fajen_approach', 'b1': 1.2035286038289252, 'k1': 11.777197150843216, 'c1': 3.208109885882151, 'c2': 0.10077453224090854, 'k2': 0.17721496598944428, 'ps': 1.0751463986115937}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][6] = {'name': 'fajen_approach', 'b1': 1.067814211145924, 'k1': 3.442460688134991, 'c1': 8.719678312497393, 'c2': 0.4467038517464913, 'k2': 0.5873392854183639, 'ps': 1.31109726329129}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][7] = {'name': 'fajen_approach', 'b1': 1.684749413674761, 'k1': 3.66442070723263, 'c1': 5.8852457414157255, 'c2': 0.2952550401219027, 'k2': 0.5739458333379727, 'ps': 1.156007882055671}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][8] = {'name': 'fajen_approach', 'b1': 1.2054388522573258, 'k1': 3.4363263877000607, 'c1': 1.9210228798200861, 'c2': 0.4370998404967321, 'k2': 0.4014795257504464, 'ps': 1.1563239895464452}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][9] = {'name': 'fajen_approach', 'b1': 1.1936413326262518, 'k1': 1.5253260080485234, 'c1': 4.71969731760836, 'c2': 0.7389384145132375, 'k2': 0.37612127377107535, 'ps': 1.1967108147865295}
approaches['Bai_movObst1']['fajen_approach']['differential_evolution'][10] = {'name': 'fajen_approach', 'b1': 1.2398029027292385, 'k1': 14.369326267732422, 'c1': 0.7651801554578793, 'c2': 0.13957426997817068, 'k2': 0.4360750101027109, 'ps': 1.056705515676128}

approaches['Fajen_steer1a'] = {'fajen_approach': {}, 'fajen_approach2': {}, 'acceleration_approach': {}}

approaches['Fajen_steer1a']['fajen_approach']['differential_evolution'] = {}
approaches['Fajen_steer1a']['fajen_approach']['differential_evolution'][-1] = {'name': 'fajen_approach', 'b1':2.01938384, 'k1':4.90527274, 'c1':2.96094879, 'c2':0.50896525, 'k2':1.61216734}

approaches['Fajen_steer1a']['fajen_approach2']['differential_evolution'] = {}
approaches['Fajen_steer1a']['fajen_approach2']['differential_evolution'][-1] = {'name': 'fajen_approach2', 'b1':2.04992354, 'k1':2.85641543, 'c1':0.54294928, 'c2':0.73857217, 'b2':3.89580222, 'k2':5.04511601}

approaches['Fajen_steer1a']['acceleration_approach']['differential_evolution'] = {}
approaches['Fajen_steer1a']['acceleration_approach']['differential_evolution'][-1] = {'name': 'acceleration_approach', 'k': 0.9220813483746707}


# Parameter boundaries for fitting

model_bounds = {}
model_bounds['fajen_approach'] = [(0, 10), (0, 20), (0, 5), (0, 5), (0, 5)]
model_bounds['fajen_approach2'] = [(0, 10), (0, 20), (0, 5), (0, 5), (0, 10), (0, 20)]
model_bounds['acceleration_approach'] = [(0, 10)]
model_bounds['jerk_approach'] = [(0, 100), (0, 50)]
model_bounds['cohen_avoid'] = [(0, 50), (0, 800), (0.1, 20), (0.1, 10), (0, 50), (0, 800), (0.1, 10), (0.1, 10)]
model_bounds['cohen_avoid_heading'] = [(0, 50), (0, 800), (0.1, 20), (0.1, 10)]
model_bounds['cohen_avoid2'] = [(0, 50), (0, 800), (0.1, 10), (1, 30), (0, 50), (0, 800), (0.1, 10), (1, 30)]
model_bounds['cohen_avoid3'] = [(0, 100), (0.1, 10), (1, 30), (0, 100), (0.1, 10), (1, 30)]
model_bounds['cohen_avoid4'] = [(0, 100), (0.1, 10), (1, 30), (0, 100), (0.1, 10), (1, 30)]
model_bounds['cohen_avoid4_heading'] = [(0, 100), (0.1, 10), (1, 30)]
model_bounds['cohen_avoid4_thres'] = [(0, 100), (0.1, 10), (1, 30), (0, 100), (0.1, 10), (1, 30), (0, 0.1)]
model_bounds['perpendicular_avoid'] = [(0, 50), (0.0001, 50)]
model_bounds['perpendicular_avoid2'] = [(0, 50), (0.0001, 20)]


# Fitted avoidance model
avoidances = {}
avoidances['Cohen_movObst1'] = {'cohen_avoid': {},
                                'cohen_avoid2': {},
                                'cohen_avoid3': {},
                                'cohen_avoid4': {},
                                'perpendicular_avoid': {},
                                'perpendicular_avoid2': {},
                                'cohen_avoid_heading': {},
                                'cohen_avoid4_heading': {}}
avoidances['Cohen_movObst2'] = {'cohen_avoid': {},
                                'cohen_avoid2': {},
                                'cohen_avoid3': {},
                                'cohen_avoid4': {},
                                'perpendicular_avoid': {},
                                'perpendicular_avoid2': {},
                                'cohen_avoid_heading': {},
                                'cohen_avoid4_heading': {}}
avoidances['Bai_movObst1'] = {'cohen_avoid': {},
                              'cohen_avoid2': {},
                              'cohen_avoid3': {},
                              'cohen_avoid4': {},
                              'perpendicular_avoid': {},
                              'perpendicular_avoid2': {},
                              'cohen_avoid_heading': {},
                              'cohen_avoid4_heading': {}}

avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'] = {}
# 0.19077652321401395 order_accuracy 0.84321608040201
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid', 'b1': 0.8093007767780838, 'k1': 374.53835960146876, 'c5': 11.763605885702065, 'c6': 1.0266683672796317, 'b2': 0.0, 'k2': 593.7861081468136, 'c7': 9.55544925847897, 'c8': 2.05798599654717}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][0] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][1] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][2] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][3] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][4] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][6] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][7] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][9] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][10] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][11] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][12] = {}
avoidances['Cohen_movObst1']['cohen_avoid']['differential_evolution'][13] = {}

avoidances['Cohen_movObst1']['cohen_avoid_heading']['differential_evolution'] = {}
avoidances['Cohen_movObst1']['cohen_avoid_heading']['differential_evolution']['-1dpsi_onset'] = {'name': 'cohen_avoid_heading', 'b1': 1.5375301722264427, 'k1': 380.14137984134635, 'c5': 11.369262765614588, 'c6': 1.0349190438337557}

avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'] = {}
# 0.1901285989843772 order_accuracy 0.8412060301507538
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid2', 'b1': 1.2349633158265405, 'k1': 128.80418613005364, 'c5': 9.75076347435121, 'c6': 8.160165995185125, 'b2': 0.0, 'k2': 9.751939026529449, 'c7': 10.0, 'c8': 20.31149333769062}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][0] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][1] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][2] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][3] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][4] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][6] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][7] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][9] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][10] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][11] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][12] = {}
avoidances['Cohen_movObst1']['cohen_avoid2']['differential_evolution'][13] = {}

avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'] = {}
# 0.1806628219285373 order_accuracy 0.842211055276382
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid3', 'k1': 34.354529381388424, 'c5': 9.724597060360962, 'c6': 28.76965929857869, 'k2': 24.141604085458468, 'c7': 9.427836309508633, 'c8': 27.937064770015027}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][0] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][1] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][2] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][3] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][4] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][6] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][7] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][9] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][10] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][11] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][12] = {}
avoidances['Cohen_movObst1']['cohen_avoid3']['differential_evolution'][13] = {}

avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'] = {}
# 0.18046731918746603 order_accuracy 0.8381909547738694
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid4', 'k1': 3.239795322411539, 'c5': 1.4547076116887092, 'c6': 7.3701320561617045, 'k2': 9.254270206225431, 'c7': 4.973658668545465, 'c8': 3.407671408524374}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][0] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][1] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][2] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][3] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][4] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][6] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][7] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][9] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][10] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][11] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][12] = {}
avoidances['Cohen_movObst1']['cohen_avoid4']['differential_evolution'][13] = {}

avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution']['-1obst_onset'] =  {'name': 'perpendicular_avoid', 'k': 16.747579485419234, 'c': 37.86594245245854}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][0] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][1] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][2] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][3] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][4] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][6] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][7] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][9] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][10] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][11] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][12] = {}
avoidances['Cohen_movObst1']['perpendicular_avoid']['differential_evolution'][13] = {}

avoidances['Cohen_movObst1']['cohen_avoid4_heading']['differential_evolution'] = {}
avoidances['Cohen_movObst1']['cohen_avoid4_heading']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid4_heading', 'k1': 12.160262507168044, 'c5': 1.1741122490946225, 'c6': 1.703848842425795}

avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'] = {}
# 0.2492772784952851 order_accuracy 0.7897142857142857
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid', 'b1': 3.363638816353943, 'k1': 95.09517044816117, 'c5': 11.400136527121575, 'c6': 0.48808376780737783, 'b2': 0.0, 'k2': 594.7235062798342, 'c7': 4.535696669262642, 'c8': 3.777451780362486}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][0] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][1] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][2] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][3] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][4] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][6] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][7] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][9] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][10] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][11] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][12] = {}
avoidances['Cohen_movObst2']['cohen_avoid']['differential_evolution'][13] = {}

avoidances['Cohen_movObst2']['cohen_avoid_heading']['differential_evolution'] = {}
avoidances['Cohen_movObst2']['cohen_avoid_heading']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid_heading', 'b1': 4.293726415485055, 'k1': 367.7714285222081, 'c5': 14.196324039474277, 'c6': 0.8005514126692047}

avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'] = {}
# 0.24663529940417053 order_accuracy 0.7851428571428571
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid2', 'b1': 3.3485438586491, 'k1': 119.59655224362285, 'c5': 8.320041684111693, 'c6': 9.59390592684502, 'b2': 0.0, 'k2': 16.022521094515888, 'c7': 3.714647768807139, 'c8': 3.971298762659632}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][0] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][1] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][2] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][3] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][4] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][6] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][7] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][9] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][10] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][11] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][12] = {}
avoidances['Cohen_movObst2']['cohen_avoid2']['differential_evolution'][13] = {}

avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'] = {}
# 0.24306915061740852 order_accuracy 0.7908571428571428
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid3', 'k1': 27.015824466499474, 'c5': 6.439764681301262, 'c6': 28.71370803318366, 'k2': 23.09740785255075, 'c7': 3.6069831344642873, 'c8': 13.659355893212165}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][0] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][1] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][2] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][3] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][4] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][6] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][7] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][9] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][10] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][11] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][12] = {}
avoidances['Cohen_movObst2']['cohen_avoid3']['differential_evolution'][13] = {}

avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'] = {}
# 0.24684128447605852 order_accuracy 0.7885714285714286
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution']['-1obst_onset'] = {'name': 'cohen_avoid4', 'k1': 2.0736814583750123, 'c5': 0.277869777057664, 'c6': 11.55562093696106, 'k2': 0.695174380649082, 'c7': 5.707392197706118, 'c8': 7.709406732831272}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][0] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][1] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][2] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][3] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][4] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][6] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][7] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][9] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][10] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][11] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][12] = {}
avoidances['Cohen_movObst2']['cohen_avoid4']['differential_evolution'][13] = {}

avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][-1] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][0] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][1] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][2] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][3] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][4] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][6] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][7] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][9] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][10] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][11] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][12] = {}
avoidances['Cohen_movObst2']['perpendicular_avoid']['differential_evolution'][13] = {}


avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution'] = {}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution'][-1] = {'name': 'cohen_avoid', 'b1': 5.087677717932208, 'k1': 192.0253975376635, 'c5': 17.316665439793844, 'c6': 0.26842504263381034, 'b2': 0.0, 'k2': 85.72517058043863, 'c7': 6.641013090337525, 'c8': 0.9339496836282939}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['-1dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 4.642525322134875, 'k1': 502.8560291766539, 'c5': 14.952434368653059, 'c6': 0.33097327028944756, 'b2': 0.0, 'k2': 15.710653241018438, 'c7': 9.450938176626648, 'c8': 0.25055228278197594}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['0dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.09098388731176317, 'k1': 594.214817726885, 'c5': 14.420477841420471, 'c6': 0.4161048020810011, 'b2': 0.019308135992638517, 'k2': 18.060397499673513, 'c7': 9.83465625606361, 'c8': 0.3553896928536771}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['1dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 4.286077744185893, 'k1': 727.8582370725359, 'c5': 17.38054337535812, 'c6': 0.3075579600035621, 'b2': 26.521937704437867, 'k2': 220.47422098786146, 'c7': 9.908950282708284, 'c8': 0.1283507795554835}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['2dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 1.6096275125822497, 'k1': 613.6807662063433, 'c5': 17.308136594011355, 'c6': 0.43420482638338254, 'b2': 0.02340407698077964, 'k2': 12.611600528356654, 'c7': 9.692476812035704, 'c8': 0.11591090603771281}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['3dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.0, 'k1': 697.3902818764312, 'c5': 14.728659190712316, 'c6': 0.4188734383230471, 'b2': 0.0, 'k2': 680.1708331226496, 'c7': 4.103120532388748, 'c8': 6.832310631171471}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['4dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.0, 'k1': 696.9735057216063, 'c5': 17.145755037206705, 'c6': 0.4724780032302319, 'b2': 0.16716708132649633, 'k2': 9.500611256617336, 'c7': 8.626459990024793, 'c8': 0.10785949619231687}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['6dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.08569450388711657, 'k1': 776.4353042243292, 'c5': 7.9023566506786365, 'c6': 0.7958328056430289, 'b2': 0.0, 'k2': 9.977686900838448, 'c7': 9.859261011900568, 'c8': 0.16405341038141075}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['7dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 6.449656444063491, 'k1': 786.743884864701, 'c5': 14.625057676606016, 'c6': 0.5337570850328899, 'b2': 0.008554522575574198, 'k2': 8.650152719873528, 'c7': 9.191066039589042, 'c8': 0.10112421752913711}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['9dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.08676553607703032, 'k1': 763.6081540621236, 'c5': 10.379450176563797, 'c6': 0.490784226729024, 'b2': 0.0008102999504049691, 'k2': 28.719760744841434, 'c7': 9.85958030293486, 'c8': 0.35681688377897}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['10dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.19291842580026303, 'k1': 624.9610610564463, 'c5': 11.215312049335635, 'c6': 1.317753135366271, 'b2': 0.0, 'k2': 66.0764904362797, 'c7': 9.94900888945114, 'c8': 0.7239429170975822}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['11dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.12044806662301777, 'k1': 741.2708154309535, 'c5': 13.48554159548286, 'c6': 0.7068949850625904, 'b2': 1.2528725333346807e-05, 'k2': 33.64564835397901, 'c7': 9.869458325785175, 'c8': 0.6781422001879563}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['12dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.2417208857047588, 'k1': 687.1429371857103, 'c5': 10.700608386039523, 'c6': 0.6350009472233051, 'b2': 0.005417446968753245, 'k2': 105.47767913101376, 'c7': 9.834575029364824, 'c8': 0.5619091904104663}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['13dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.012788967699064005, 'k1': 775.6982856566221, 'c5': 11.57793441676696, 'c6': 0.9176639966606995, 'b2': 0.0036864447798379946, 'k2': 8.171884217269096, 'c7': 8.82661269821173, 'c8': 0.1639927203080436}

avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['-1dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 9.83282644978439, 'k1': 695.2122003681428, 'c5': 17.169776454025403, 'c6': 0.3076977893637865, 'b2': 0.0, 'k2': 29.364014818866437, 'c7': 9.969220735466491, 'c8': 0.3930722105146914}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['0dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.18985658718311663, 'k1': 768.2181431235703, 'c5': 15.65903461164075, 'c6': 0.44232542061321556, 'b2': 0.02283024691315745, 'k2': 32.47326155158014, 'c7': 9.977920032788267, 'c8': 0.48611667933186625}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['1dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 3.6658237234036193, 'k1': 427.11119435676824, 'c5': 19.049899438828298, 'c6': 0.18268873555478748, 'b2': 0.000762808774410928, 'k2': 21.873664798604523, 'c7': 9.976402724994355, 'c8': 0.24055355989319693}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['2dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 2.512641169031841, 'k1': 697.1267684936408, 'c5': 19.348634824395624, 'c6': 0.40928214205729313, 'b2': 0.07494804406024978, 'k2': 13.057794386377592, 'c7': 9.403341141023049, 'c8': 0.1278450724535358}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['3dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.0, 'k1': 522.0816738438788, 'c5': 13.463291460145092, 'c6': 0.41069332000272124, 'b2': 0.0, 'k2': 494.43064309115744, 'c7': 9.700730437893741, 'c8': 7.921726091151997}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['4dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.19837318027397194, 'k1': 610.273771921272, 'c5': 18.95094233086519, 'c6': 0.40065771017500335, 'b2': 0.0, 'k2': 16.23689602427115, 'c7': 9.923197079879092, 'c8': 0.2048704312170955}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['6dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.11832487254703596, 'k1': 729.3398400154548, 'c5': 8.48641562279948, 'c6': 0.7570706907698201, 'b2': 0.0036581179975837745, 'k2': 11.698848713123347, 'c7': 9.79190222571993, 'c8': 0.2003141794021037}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['7dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 3.5476362166243804, 'k1': 734.9336886514285, 'c5': 18.231152804996036, 'c6': 0.4805028209732391, 'b2': 0.0027403025294358394, 'k2': 9.309323113144778, 'c7': 9.850552659172676, 'c8': 0.1082812730114052}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['9dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.14818810678034022, 'k1': 781.2937496558328, 'c5': 12.747682504744738, 'c6': 0.4271977481801153, 'b2': 0.0, 'k2': 11.942091785526598, 'c7': 9.727304149836064, 'c8': 0.1788030442725787}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['10dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.5823472028336445, 'k1': 503.5193603999053, 'c5': 18.152125815317298, 'c6': 0.8938150562476228, 'b2': 0.00034184685620352196, 'k2': 83.76968591395519, 'c7': 9.846591786245027, 'c8': 0.7759189009965027}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['11dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.07332494807633706, 'k1': 791.883383148395, 'c5': 14.780319091114599, 'c6': 0.6766788468553773, 'b2': 0.019025820668411485, 'k2': 17.83539433671126, 'c7': 9.844258799874178, 'c8': 0.46956805297611487}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['12dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.06364920088952006, 'k1': 648.8779654096805, 'c5': 10.116220234902304, 'c6': 0.669557169597552, 'b2': 0.0, 'k2': 154.90181102933798, 'c7': 9.946422740391172, 'c8': 0.6550195734977055}
avoidances['Bai_movObst1']['cohen_avoid']['differential_evolution']['13dtheta*dpsi_onset'] = {'name': 'cohen_avoid', 'b1': 0.10573475465978198, 'k1': 777.7757049679349, 'c5': 16.289459191308428, 'c6': 0.7406720059444575, 'b2': 0.0, 'k2': 19.078200959962412, 'c7': 9.60061569310974, 'c8': 0.37752088371725884}

avoidances['Bai_movObst1']['cohen_avoid_heading']['differential_evolution'] = {}
avoidances['Bai_movObst1']['cohen_avoid_heading']['differential_evolution']['-1dpsi_onset'] = {'name': 'cohen_avoid_heading', 'b1': 9.930552269026572, 'k1': 649.82737708828, 'c5': 17.90090735259422, 'c6': 0.2844581112275794}

avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution'] = {}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution'][-1] = {'name': 'cohen_avoid2', 'b1': 0.21024238572896845, 'k1': 147.48966572415804, 'c5': 8.383082639840252, 'c6': 9.33498860697201, 'b2': 0.0, 'k2': 10.751693520690063, 'c7': 9.322384316146534, 'c8': 28.99758888221377}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['-1dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 3.31660936296053, 'k1': 265.24585586337037, 'c5': 9.737473661386517, 'c6': 11.609850708869496, 'b2': 0.0, 'k2': 20.44218194006397, 'c7': 9.879816352314704, 'c8': 14.825253227271302}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['0dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 175.1266782919033, 'c5': 10.0, 'c6': 18.33415650787958, 'b2': 0.0, 'k2': 91.016945427529, 'c7': 10.0, 'c8': 1.9715456748889322}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['1dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.23774232340550938, 'k1': 643.8538921265106, 'c5': 9.998435174189762, 'c6': 4.905671679696134, 'b2': 46.72533879566697, 'k2': 544.8894238353804, 'c7': 9.811920038933005, 'c8': 27.91549580051953}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['2dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.002938733608002621, 'k1': 214.16348025793778, 'c5': 9.961829012195631, 'c6': 7.7035501079480815, 'b2': 0.021319251993631938, 'k2': 14.529557889599491, 'c7': 9.127140739102565, 'c8': 29.91042755394572}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['3dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 129.23505284474103, 'c5': 9.740837661170312, 'c6': 24.367943314000456, 'b2': 0.0, 'k2': 3.5787146771317238, 'c7': 9.93898866658417, 'c8': 26.28690152935446}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['4dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 6.169954202910075e-05, 'k1': 99.25895388422376, 'c5': 10.0, 'c6': 19.185975559872755, 'b2': 0.1296322695754799, 'k2': 17.28169813593548, 'c7': 10.0, 'c8': 28.98394764732919}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['6dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.03225198237045449, 'k1': 749.3831516627911, 'c5': 6.522006737858936, 'c6': 1.2946550503745753, 'b2': 0.0, 'k2': 9.492226066936016, 'c7': 9.602084905669223, 'c8': 25.97567355112247}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['7dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 1.7284776051732154, 'k1': 397.7936906510534, 'c5': 9.935164979961458, 'c6': 4.418643087063296, 'b2': 0.0, 'k2': 12.556601597981311, 'c7': 10.0, 'c8': 29.08141361328366}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['9dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 422.0966058322661, 'c5': 9.597659866207113, 'c6': 10.496342339849003, 'b2': 0.0, 'k2': 13.17225641255577, 'c7': 9.992336189654374, 'c8': 29.832686287082993}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['10dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.06580585899366527, 'k1': 165.60852909471114, 'c5': 9.349269989304162, 'c6': 1.6809506190556691, 'b2': 0.0, 'k2': 17.737310048822405, 'c7': 9.935282430616201, 'c8': 13.725758324019008}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['11dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.05514187083939052, 'k1': 199.32190059217774, 'c5': 9.900797701500084, 'c6': 6.002574833039962, 'b2': 0.0, 'k2': 15.445316989508498, 'c7': 9.397547267573174, 'c8': 5.470934827559164}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['12dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.44906115653892903, 'k1': 350.9784763226569, 'c5': 9.036352589003473, 'c6': 5.904372992142564, 'b2': 0.0, 'k2': 21.405213905996792, 'c7': 9.77777949526191, 'c8': 28.449071470452196}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['13dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 750.8685652637598, 'c5': 9.855056103095956, 'c6': 1.1383964828262607, 'b2': 0.0, 'k2': 9.393982114456499, 'c7': 8.610575718807628, 'c8': 20.53422264880004}

avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['-1dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 3.7458602358806234, 'k1': 358.78415514040466, 'c5': 9.653751371830825, 'c6': 7.714053696476542, 'b2': 0.0, 'k2': 95.26312473727977, 'c7': 9.86777088546365, 'c8': 2.454014366651849}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['0dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 222.85617483052496, 'c5': 10.0, 'c6': 13.308330883823329, 'b2': 0.0, 'k2': 15.014345887233024, 'c7': 10.0, 'c8': 14.178338393742314}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['1dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.06983956611042029, 'k1': 241.12855746731228, 'c5': 10.0, 'c6': 13.238575581795859, 'b2': 0.0, 'k2': 18.626660392469564, 'c7': 9.936463340340955, 'c8': 26.17848234726081}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['2dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.1130312132175768, 'k1': 175.97678513779854, 'c5': 9.873251074374732, 'c6': 9.004702320951774, 'b2': 0.010742516129145934, 'k2': 16.182299032371102, 'c7': 9.908997674899874, 'c8': 29.570897009517733}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['3dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 198.07350286192263, 'c5': 10.0, 'c6': 14.17543704295502, 'b2': 0.0, 'k2': 0.0, 'c7': 10.0, 'c8': 13.558827135219197}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['4dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 176.16484491061942, 'c5': 10.0, 'c6': 9.767923974435131, 'b2': 0.0, 'k2': 14.894593757778344, 'c7': 10.0, 'c8': 30.0}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['6dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 680.8929684514264, 'c5': 7.820980873112664, 'c6': 1.7581290852332434, 'b2': 0.0, 'k2': 30.435032695299842, 'c7': 10.0, 'c8': 7.480418562443733}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['7dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 1.4941296122656942, 'k1': 374.98250510151854, 'c5': 9.940506606997973, 'c6': 4.51857369607572, 'b2': 0.07470147913936, 'k2': 12.934188644031615, 'c7': 9.662915682330164, 'c8': 27.654693755668635}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['9dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0, 'k1': 462.0577423083397, 'c5': 10.0, 'c6': 9.614002251469499, 'b2': 0.0, 'k2': 11.548492616374775, 'c7': 9.87012322151855, 'c8': 26.719325384083103}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['10dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.4217557416924729, 'k1': 33.12614781219072, 'c5': 9.231008499148828, 'c6': 10.316909652310017, 'b2': 0.0, 'k2': 26.619830317972994, 'c7': 9.548263961563215, 'c8': 8.851646559446351}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['11dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.008313835280016871, 'k1': 369.0031777087851, 'c5': 10.0, 'c6': 3.2718748531167154, 'b2': 0.0, 'k2': 14.03339522337245, 'c7': 9.828137961115221, 'c8': 7.927723343459421}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['12dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.1651501823548781, 'k1': 256.4591292989368, 'c5': 9.536360072590853, 'c6': 8.460487741706952, 'b2': 0.0, 'k2': 77.24443637753134, 'c7': 9.987404406072846, 'c8': 6.727698417507627}
avoidances['Bai_movObst1']['cohen_avoid2']['differential_evolution']['13dtheta*dpsi_onset'] = {'name': 'cohen_avoid2', 'b1': 0.0003605229673991114, 'k1': 469.42853319309495, 'c5': 9.929239077971571, 'c6': 2.026311616457178, 'b2': 0.0, 'k2': 10.100725030847736, 'c7': 9.142731007026795, 'c8': 22.356214477593863}

avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution'] = {}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution'][-1] = {'name': 'cohen_avoid3', 'k1': 55.796632495340106, 'c5': 10.0, 'c6': 27.837737350622703, 'k2': 26.03429427752777, 'c7': 10.0, 'c8': 28.384796742197093}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['-1dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 93.42068415612331, 'c5': 9.923230152411703, 'c6': 20.077941579021086, 'k2': 22.321805247910984, 'c7': 9.834536869882928, 'c8': 28.10117113055228}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['0dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 100.0, 'c5': 9.170370768359117, 'c6': 30.0, 'k2': 27.761243012133942, 'c7': 9.36611455321723, 'c8': 28.211865589483548}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['1dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 98.92106206509318, 'c5': 8.239279826284848, 'c6': 29.947602076667653, 'k2': 44.574384119110796, 'c7': 9.80716407313302, 'c8': 27.465950133315683}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['2dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 92.18198345141445, 'c5': 10.0, 'c6': 12.301947764630182, 'k2': 41.30595529614932, 'c7': 9.920427145491614, 'c8': 30.0}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['3dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.50722575237785, 'c5': 8.195879067098153, 'c6': 29.99146120529184, 'k2': 23.115887914435007, 'c7': 9.922318572155229, 'c8': 15.235817853891115}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['4dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 84.49415316065652, 'c5': 10.0, 'c6': 21.508157789627727, 'k2': 35.55772126649435, 'c7': 10.0, 'c8': 30.0}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['6dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.61512552050414, 'c5': 5.519868939674103, 'c6': 10.651540267418678, 'k2': 26.571830648537357, 'c7': 9.976620705304294, 'c8': 14.425950895311036}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['7dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.61204238903149, 'c5': 9.605095878910316, 'c6': 17.762929240165356, 'k2': 12.853914032733048, 'c7': 9.444711764616576, 'c8': 28.823845132558432}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['9dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 98.84574764945233, 'c5': 5.6930476038298305, 'c6': 29.96849256350834, 'k2': 32.982746038232044, 'c7': 7.13711575851509, 'c8': 14.213532349986568}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['10dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 95.21677444349466, 'c5': 9.814965586601819, 'c6': 2.856501507529275, 'k2': 49.14740523381079, 'c7': 9.550278074361863, 'c8': 13.590045790188759}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['11dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 97.30880177662193, 'c5': 9.99998140815632, 'c6': 11.627905865893611, 'k2': 10.227845710426088, 'c7': 9.706120265224918, 'c8': 29.68678326461985}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['12dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.52045787210625, 'c5': 6.740547031313627, 'c6': 26.166432629084515, 'k2': 55.98189689377361, 'c7': 9.931444543688006, 'c8': 26.453708905727122}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['13dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.49076483357302, 'c5': 8.340167383259022, 'c6': 7.076312526779212, 'k2': 17.679382605033915, 'c7': 9.270702107449488, 'c8': 28.651245059754366}

avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['-1dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 82.03700045827495, 'c5': 10.0, 'c6': 23.32713083634937, 'k2': 24.169946013376087, 'c7': 10.0, 'c8': 26.501894060385826}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['0dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.78387565202497, 'c5': 9.245582034572168, 'c6': 29.96711072480562, 'k2': 28.600291811484112, 'c7': 9.545712231694692, 'c8': 28.652537451287422}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['1dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.73885666470636, 'c5': 8.628857793013552, 'c6': 30.0, 'k2': 42.477400245111404, 'c7': 10.0, 'c8': 29.76443724713188}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['2dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 98.7886459124899, 'c5': 9.99999999, 'c6': 11.027835708015182, 'k2': 37.39876052724272, 'c7': 9.227923580956714, 'c8': 29.832523317498808}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['3dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 100.0, 'c5': 8.77691722395675, 'c6': 30.0, 'k2': 0.9853635892873728, 'c7': 9.992171396674713, 'c8': 1.6602835778309664}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['4dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 67.96597316637333, 'c5': 9.993383601477843, 'c6': 26.379696636331786, 'k2': 30.139708445335064, 'c7': 9.761908941642965, 'c8': 29.863854301170566}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['6dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 98.28760063102996, 'c5': 5.636121160793337, 'c6': 10.585223031679842, 'k2': 17.594429508378756, 'c7': 9.737126042708075, 'c8': 17.984146316541608}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['7dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 98.82140781057547, 'c5': 10.0, 'c6': 18.644726187649454, 'k2': 14.492007117264992, 'c7': 10.0, 'c8': 30.0}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['9dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 98.97318507632448, 'c5': 6.154524713638621, 'c6': 30.0, 'k2': 22.11702932272038, 'c7': 6.64943263126249, 'c8': 20.681907653192543}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['10dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 67.46392428513406, 'c5': 9.961233161222744, 'c6': 4.830028333578063, 'k2': 74.87698078852308, 'c7': 9.961294835697224, 'c8': 9.336757838294567}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['11dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 98.09010056244082, 'c5': 10.0, 'c6': 12.224271506965373, 'k2': 75.90461656806035, 'c7': 10.0, 'c8': 3.3220326039521373}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['12dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.20527088874849, 'c5': 7.300715308817116, 'c6': 29.12421546426165, 'k2': 82.5281038751876, 'c7': 9.75888338402696, 'c8': 16.37236633000221}
avoidances['Bai_movObst1']['cohen_avoid3']['differential_evolution']['13dtheta*dpsi_onset'] = {'name': 'cohen_avoid3', 'k1': 99.87670942756237, 'c5': 10.0, 'c6': 10.020685793539956, 'k2': 16.310815048462583, 'c7': 8.321947369055646, 'c8': 27.571784323467185}

avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution'] = {}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution'][-1] = {'name': 'cohen_avoid4', 'k1': 1.2371671664563857, 'c5': 0.33688315045398826, 'c6': 27.933552305969986, 'k2': 5.2676318261977535, 'c7': 3.841131435146948, 'c8': 3.753622569591886}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['-1dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 35.66006018987271, 'c5': 6.211312647402919, 'c6': 2.8829899279613884, 'k2': 26.297504220855956, 'c7': 9.950133508470925, 'c8': 1.8318341573512125}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['0dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 15.334011045146003, 'c5': 6.995452941050658, 'c6': 13.80821757955234, 'k2': 2.051952910314153, 'c7': 6.04780648191303, 'c8': 26.18331334743836}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['1dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 35.181704357080534, 'c5': 6.796070699190349, 'c6': 6.499211898407426, 'k2': 3.9437361842692633, 'c7': 7.365866933892889, 'c8': 23.19262641589863}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['2dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 37.16078043858363, 'c5': 9.553449144635826, 'c6': 2.813621732590878, 'k2': 3.5671566957723093, 'c7': 7.964565946353126, 'c8': 29.86540115458712}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['3dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 94.96264512378701, 'c5': 6.448118997946248, 'c6': 2.63883731515896, 'k2': 17.928625162366732, 'c7': 9.984122552978151, 'c8': 3.0091823787583265}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['4dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 24.226189458790888, 'c5': 8.508228264258815, 'c6': 5.647979534040848, 'k2': 8.58294815578727, 'c7': 9.810994759003144, 'c8': 9.814069622151111}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['6dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 62.64596485085455, 'c5': 1.9540787375000335, 'c6': 1.082124870817828, 'k2': 8.4058271580466, 'c7': 9.07092073367395, 'c8': 3.256081220920679}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['7dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 27.194651341126715, 'c5': 5.144692487664247, 'c6': 3.194884892962395, 'k2': 1.3212229978367167, 'c7': 9.363907130292874, 'c8': 28.007034386826636}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['9dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 63.52931579966246, 'c5': 5.407899839097372, 'c6': 4.189098303207256, 'k2': 2.0591581208222585, 'c7': 6.810996283934313, 'c8': 29.80252842971981}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['10dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 8.733285747758195, 'c5': 6.315965791378985, 'c6': 1.9237763866688844, 'k2': 10.423318223276713, 'c7': 6.192919056371341, 'c8': 3.670664837433816}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['11dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 15.778430291301751, 'c5': 4.7593266153757945, 'c6': 3.563840358757792, 'k2': 1.3677038449753898, 'c7': 3.8011981880702703, 'c8': 8.756441634899552}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['12dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 35.93982976665939, 'c5': 4.29598545887183, 'c6': 4.719161771115112, 'k2': 4.103704390701866, 'c7': 6.365440570090229, 'c8': 22.88124419348531}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['13dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 42.15334598965811, 'c5': 4.657425527632163, 'c6': 1.0282907256210763, 'k2': 0.6796768951547572, 'c7': 1.4196317296042222, 'c8': 29.38104446906016}

avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['-1dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 6.1861420631738895, 'c5': 5.26855222266815, 'c6': 16.702063222816975, 'k2': 2.2006528003171666, 'c7': 8.104901078488547, 'c8': 21.267789382346802}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['0dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 23.28410069978652, 'c5': 7.541413374970457, 'c6': 9.247458941236914, 'k2': 4.7175172169162245, 'c7': 6.4267394146104815, 'c8': 10.644194473566953}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['1dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 30.44585471028472, 'c5': 7.639404545904846, 'c6': 8.661930586702956, 'k2': 3.3665543062121976, 'c7': 7.379662935553921, 'c8': 29.640455708000584}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['2dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 52.92245398068288, 'c5': 9.902683710759435, 'c6': 1.9592318232475125, 'k2': 2.666084046308319, 'c7': 5.285266983854481, 'c8': 28.863348201081312}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['3dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 20.93784759542478, 'c5': 6.648896487797643, 'c6': 13.403908308703683, 'k2': 7.410602603189195, 'c7': 10.0, 'c8': 1.2944245657021987}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['4dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 14.251149055667153, 'c5': 9.018830070506022, 'c6': 11.09384050143815, 'k2': 2.9357397946491166, 'c7': 7.452821386377392, 'c8': 26.41985468112714}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['6dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 40.148752480543365, 'c5': 1.7536944242084354, 'c6': 1.6607382117559002, 'k2': 1.7872227672968974, 'c7': 9.987565155403507, 'c8': 16.633331919808185}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['7dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 46.574835670273885, 'c5': 5.830395496106008, 'c6': 1.940554153931508, 'k2': 1.0738085344001123, 'c7': 6.116239258661899, 'c8': 20.948658470677216}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['9dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 51.60591827900249, 'c5': 7.040191084775717, 'c6': 6.313776325680487, 'k2': 18.259203289768926, 'c7': 7.677802352742288, 'c8': 2.626087016137817}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['10dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 12.24408499851723, 'c5': 9.487121362412683, 'c6': 2.2560466368902303, 'k2': 14.080475224809241, 'c7': 5.806499493778887, 'c8': 2.725595902251843}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['11dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 30.756172260558824, 'c5': 5.599559693282266, 'c6': 2.050319994538104, 'k2': 2.3868229728796, 'c7': 8.58611286322914, 'c8': 9.5172529430328}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['12dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 76.54034600821892, 'c5': 4.917342906635575, 'c6': 2.3016443663767485, 'k2': 8.512883387531572, 'c7': 5.919674646989444, 'c8': 9.698924260662823}
avoidances['Bai_movObst1']['cohen_avoid4']['differential_evolution']['13dtheta*dpsi_onset'] = {'name': 'cohen_avoid4', 'k1': 44.17435437010587, 'c5': 9.823766555786904, 'c6': 1.7976736974616494, 'k2': 5.15651562119465, 'c7': 2.549419532966808, 'c8': 2.838600189516801}


avoidances['Bai_movObst1']['cohen_avoid4_heading']['differential_evolution'] = {}
avoidances['Bai_movObst1']['cohen_avoid4_heading']['differential_evolution'][-1] = {'name': 'cohen_avoid4_heading', 'k1': 7.0288327450614645, 'c5': 5.699504822685455, 'c6': 15.368240959940403}

avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution']['-1dpsi_onset'] = {'name': 'perpendicular_avoid', 'k': 11.397868452412597, 'c': 8.813338864706262}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][0] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][1] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][2] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][3] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][4] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][6] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][7] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][9] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][10] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][11] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][12] = {}
avoidances['Bai_movObst1']['perpendicular_avoid']['differential_evolution'][13] = {}