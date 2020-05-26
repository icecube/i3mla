'''A dictionary of sources and functions to convert their locations to radians'''

def ra(h, m, s):
    '''Convert right ascension to radians'''
    return (h*15 + m/4 + s/240)*np.pi/180

def dec(sign, deg, m, s):
    '''Convert declination to radians'''
    return sign*(deg + m/60 + s/3600)*np.pi/180

# dict of sources
sources = {'crab_nebula':{'ra':ra(5, 34, 31.94), 'dec':dec(1, 22, 0, 52.2)},
           'txs'        :{'ra':ra(5, 9, 25.9645434784), 'dec':dec(1, 5, 41, 35.333636817)},
           'm77'        :{'ra':ra(2, 42, 40.771), 'dec':dec(-1, 0, 0, 47.84)},
           'mgro1908'   :{'ra':ra(19, 7, 54), 'dec':dec(1, 6, 16, 7)},
           'sag_a_star' :{'ra':ra(17, 45, 40.0409), 'dec':dec(-1, 29, 0, 28.118)},
           'mag_l'      :{'ra':ra(5, 23, 34.5), 'dec':dec(-1, 69, 45, 22)},
           'mag_s'      :{'ra':ra(0, 52, 44.8), 'dec':dec(-1, 72, 49, 43)},
}
