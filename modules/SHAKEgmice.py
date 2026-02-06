"""
Ground Motion Intensity Conversion Equations (GMICE) 
Class Name: GMICE

Description:
    The GMICE class provides functionality to convert ground motion measurements into various intensity measures and scales
    using Ground Motion Intensity Conversion Equations (GMICEs). The class supports a variety of conversion models, each 
    based on different studies and regional data.

Attributes:
    model (str): The conversion model to use. Default is 'Globalgmice'.
    input_value (array-like): The input values to be converted, given in cm/s^2 for PGA/PGV.
    input_type (str): The type of the input values ('PGA', 'PGV', 'MMI', 'EMS', 'MCS', 'sa_03', 'sa_1', 'sa_3').
    output_type (str): The type of the output values ('PGA', 'PGV', 'MMI', 'EMS', 'MCS', 'sa_03', 'sa_1', 'sa_3').

Models:
    - Globalgmice: Caprio et al. (2015) - A Global Relationship and Evaluation of Regional Dependency
    - Bilalandaskan14: Mustafa Bilal and Aysegul Askan (2014) - Relationships between Felt Intensity and Recorded Ground Motion Parameters
    - Zaniniandhofer19: Mariano Zanini and Lorenzo Hofer (2019) - Reversible ground motion-to-intensity conversion equations based on the EMS-98 scale
    - Fami10: Faenza and Michelini (2010) - Regression analysis of MCS intensity and ground motion parameters in Italy
    - WordenEtAl12: Worden et al. (2012) - Probabilistic Relationships between Ground‐Motion Parameters and Modified Mercalli Intensity in California
    - TianEtAl21: Tian et al. (2021) - New Ground Motion to Intensity Conversion Equations for China
    - WuEtAl03: Wu et al. (2003) - Relationship between Peak Ground Acceleration, Peak Ground Velocity, and Intensity in Taiwan
    - Atkinsonandkaka07: Atkinson and Kaka (2007) - Relationships between Felt Intensity and Instrumental Ground Motion in the Central United States and California
    - JMA2001: Khosrow T. Shabestaria and Fumio Yamazaki (2001) - A Proposal of Instrumental Seismic Intensity Scale Compatible with MMI Evaluated from Three-Component Acceleration Records

Methods:
    __init__(self, model='Globalgmice', input_value=None, input_type=None, output_type=None):
        Initializes the GMICE instance with the provided model, input value, input type, and output type.
        
    convert(self):
        Performs the conversion based on the initialized parameters and the selected model.
        
    list_gmice(self):
        Lists available GMICE models and their descriptions.

    Globalgmice(self):
        Implementation of the Globalgmice conversion logic.

    Bilalandaskan14(self):
        Implementation of the Bilalandaskan14 conversion logic.

    Zaniniandhofer19(self):
        Implementation of the Zaniniandhofer19 conversion logic.

    Fami10(self):
        Implementation of the Fami10 conversion logic.

    WordenEtAl12(self):
        Implementation of the WordenEtAl12 conversion logic.

    TianEtAl21(self):
        Implementation of the TianEtAl21 conversion logic.

    WuEtAl03(self):
        Implementation of the WuEtAl03 conversion logic.

    Atkinsonandkaka07(self):
        Implementation of the Atkinsonandkaka07 conversion logic.

    JMA2001(self):
        Implementation of the JMA2001 conversion logic.

Example Usage:
    converter = GMICE(model='Globalgmice', input_value=pga_values, input_type='PGA', output_type='MMI')
    results = converter.result

    # Alternative usage
    converter = GMICE(model='Globalgmice', input_value=pga_values, input_type='PGA', output_type='MMI').result

    # Printing available correlation functions
    GMICE().list_gmice()

Author:
    PLH
Date:
    May 01, 2024
Version:
    24.1
"""

import numpy as np

class GMICE:

    """
    A class to convert ground motion measurements into different intensity measures and scales using Ground Motion Intensity Conversion Equations (GMICEs)
    Supports various conversion models.

    Attributes:
        model (str): The conversion model to use. Default is 'Globalgmice'.
        input_value (array-like): The input values to be converted, given in cm/s^2 for PGA/PGV.
        input_type (str): The type of the input values ('PGA', 'PGV', or 'MMI'/ 'EMS' / 'MCS').
        output_type (str): The type of the output values ('PGA', 'PGV', or 'MMI'/ 'EMS' / 'MCS').
    Models: 
        Globalgmice : Caprio et al. (2015). Ground Motion to Intensity Conversion Equations (GMICEs): A Global Relationship and Evaluation of Regional Dependency 
        Bilalandaskan14: Mustafa Bilal and Aysegul Askan (2014). Relationships between Felt Intensity and Recorded Ground Motion Parameters for 
        Zaniniandhofer19: Mariano Angelo Zaninia (2019). Lorenzo Hofera and Flora Faleschinia (). Reversible ground motion-to-intensity conversion equations based on the EMS-98 scale
        Fami10: Faenza and Michelini (2010). Regression analysis of MCS intensity and ground motion parameters in Italy and its application in ShakeMap
        WordenEtAl12: Worden et al. (2012). Probabilistic Relationships between Ground‐Motion Parameters and Modified Mercalli Intensity in California
        TianEtAl21: Tian et al. (2021). New Ground Motion to Intensity Conversion Equations for China
        WuEtAl03: Wu et al. (2003). Relationship between Peak Ground Acceleration, Peak Ground Velocity, and Intensity in Taiwan
        Atkinsonandkaka07: Atkinson and Kaka (2007). Relationships between Felt Intensity and Instrumental Ground Motion in the Central United States and California
        
    Methods:
        convert(): Performs the conversion based on the initialized parameters.
        
    Example usage: 
        converter = GMICE(model='Globalgmice', input_value=pga_values, input_type='PGA', output_type='MMI')
        # Access results
        results = converter.result
        
        # or
        converter = GMICE(model='Globalgmice', input_value=pga_values, input_type='PGA', output_type='MMI').result
        
        # printing available corolation fucntions 
        gmice().list_gmice()


    © SHAKEmaps version 25.2.3
    """
    
    
    def __init__(self, model='Globalgmice', input_value=None, input_type=None, output_type=None):
        self.model = model
        self.input_value = np.array(input_value, ndmin=1)  # Ensures input_value is always an array
        self.input_type = input_type
        self.output_type = output_type
        # Directly perform the conversion upon initialization
        self.result = self.convert() if input_value is not None else None

    def convert(self):
        if self.model == 'Globalgmice':
            return self.Globalgmice()
        elif self.model == 'Bilalandaskan14':
            return self.Bilalandaskan14()
        elif self.model == 'Zaniniandhofer19':
            return self.Zaniniandhofer19()
        elif self.model == 'Fami10':
            return self.Fami10()
        elif self.model== 'WordenEtAl12':
            return self.WordenEtAl12()
        elif self.model== 'TianEtAl21':
            return self.TianEtAl21()
        elif self.model== 'WuEtAl03':
            return self.WuEtAl03()
        elif self.model== 'Atkinsonandkaka07':
            return self.Atkinsonandkaka07()
        elif self.model== 'JMA2001':
            return self.JMA2001()
        else:
            raise ValueError("Invalid model selected")
        
        
    # Implementation of the Globalgmice conversion logic
    def Globalgmice(self):

        input_value = np.array(self.input_value)

        # PGA to MMI
        if self.input_type == 'PGA' and self.output_type == 'MMI':
            return np.where(input_value < 0, 0,
                            np.where(input_value <= 50, 
                                     2.27 + 1.647 * np.log10(input_value), 
                                     -1.361 + 3.822 * np.log10(input_value)))

        # PGV to MMI
        if self.input_type == 'PGV' and self.output_type == 'MMI':
            return np.where(input_value < 0, 0,
                            np.where(input_value <= 2.5, 
                                     4.424 + 1.589 * np.log10(input_value), 
                                     4.018 + 2.671 * np.log10(input_value)))

        # MMI to PGA
        if self.input_type == 'MMI' and self.output_type == 'PGA':
            return np.where(input_value < 5.132463357, 
                            10 ** ((input_value - 2.27) / 1.647), 
                            10 ** ((input_value + 1.361) / 3.822))

        # MMI to PGV
        if self.input_type == 'MMI' and self.output_type == 'PGV':
            return np.where(input_value < 5.056326674, 
                            10 ** ((input_value - 4.424) / 1.589), 
                            10 ** ((input_value - 4.018) / 2.671))

        raise ValueError("Invalid input and output type combination.")
        
    # Implementation of the bilal14 conversion logic
    def Bilalandaskan14(self):

        input_value = np.array(self.input_value)

        # PGA to MMI
        if self.input_type == 'PGA' and self.output_type == 'MMI':
            mmi = np.where(input_value < 0, 0, 0.132 + 3.884 * np.log10(input_value))
            return mmi

        # PGV to MMI
        if self.input_type == 'PGV' and self.output_type == 'MMI':
            mmi = np.where(input_value < 0, 0, 2.673 + 4.340 * np.log10(input_value))
            return mmi

        # MMI to PGA
        if self.input_type == 'MMI' and self.output_type == 'PGA':
            pga = np.where(input_value < 0, 0, (input_value - 0.132) / 3.884)
            pga = 10 ** pga
            return pga

        # MMI to PGV
        if self.input_type == 'MMI' and self.output_type == 'PGV':
            pgv = np.where(input_value < 0, 0, (input_value - 2.673) / 4.340)
            pgv = 10 ** pgv
            return pgv

        raise ValueError("Invalid input and output type combination.")

    # Implementation of the Zaniniandhofer19 conversion logic
    def Zaniniandhofer19(self):
        input_value = np.array(self.input_value)

        # PGA to EMS
        if self.input_type == 'PGA' and self.output_type == 'EMS':
            ems = np.where(input_value <= 0, 0, 2.03 + 2.28 * np.log10(input_value))
            return ems

        # PGV to EMS
        if self.input_type == 'PGV' and self.output_type == 'EMS':
            ems = np.where(input_value < 0, 0, 4.16 + 1.62 * np.log10(input_value))
            return ems

        # EMS to PGA
        if self.input_type == 'EMS' and self.output_type == 'PGA':
            pga = np.where(input_value < 0, 0, (input_value - 2.03) / 2.28)
            pga = 10 ** pga
            return pga

        # EMS to PGV
        if self.input_type == 'EMS' and self.output_type == 'PGV':
            pgv = np.where(input_value < 0, 0, (input_value - 4.16) / 1.62)
            pgv = 10 ** pgv
            return pgv

        raise ValueError("Invalid input and output type combination.")
    
    # Implementation of the Fami10 conversion logic
    def Fami10(self):
        input_value = np.array(self.input_value)

        # PGA to MCS
        if self.input_type == 'PGA' and self.output_type == 'MCS':
            mcs = np.where(input_value <= 0, 0, 1.68 + 2.58 * np.log10(input_value))
            return mcs

        # PGV to MCS
        if self.input_type == 'PGV' and self.output_type == 'MCS':
            mcs = np.where(input_value < 0, 0, 2.35 * np.log10(input_value) + 5.11)
            return mcs

        # MCS to PGA
        if self.input_type == 'MCS' and self.output_type == 'PGA':
            pga = np.where(input_value < 0, 0, (input_value - 1.68) / 2.58)
            pga = 10 ** pga
            return pga

        # MCS to PGV
        if self.input_type == 'MCS' and self.output_type == 'PGV':
            pgv = np.where(input_value < 0, 0, (input_value - 5.11) / 2.35)
            pgv = 10 ** pgv
            return pgv

        raise ValueError("Invalid input and output type combination.")
    
    
    
    # Implementation of the WordenEtAl12 conversion logic
    def WordenEtAl12(self):

        input_value = np.array(self.input_value)
            
        # MMI to PGM 
        if self.input_type == 'MMI' and self.output_type in ['PGA', 'PGV','sa_03','sa_1','sa_3']:
            
            # Define the coefficients for MMI to PGA/PGV conversion
            if self.output_type == 'PGA':
                c1, c2, c3, c4, t1,t2 = 1.78, 1.55, -1.60, 3.70, 1.57,4.22
            elif self.output_type == 'PGV':  # PGV
                c1, c2, c3, c4, t1,t2 = 3.78, 1.47, 2.89, 3.16, 0.53,4.56
            elif self.output_type == 'sa_03':
                c1, c2, c3, c4, t1,t2 = 1.26,1.69,-4.15,4.14,2.21,4.99
            elif self.output_type == 'sa_1':
                c1, c2, c3, c4, t1,t2 = 2.5,1.51,0.20,2.90,1.65,4.98
            elif  self.output_type == 'sa_3':
                c1, c2, c3, c4, t1,t2 = 3.81,1.17,1.99,3.01,0.99,4.96
            else:
                raise ValueError("Invalid input and output type combination.")


            # Inverse conversion from MMI to PGA or PGV
            return np.where(input_value <= t2,
                            10 ** ((input_value - c1) / c2),
                            10 ** ((input_value - c3) / c4))
        # PGM to MMI 
        elif self.input_type in ['PGA', 'PGV','sa_03','sa_1','sa_3'] and self.output_type == 'MMI':
            if self.input_type == 'PGA':
                c1, c2, c3, c4, t1,t2 = 1.78, 1.55, -1.60, 3.70, 1.57,4.22
            elif self.input_type == 'PGV' :  # PGV
                c1, c2, c3, c4, t1,t2 = 3.78, 1.47, 2.89, 3.16, 0.53,4.56
            elif self.output_type == 'sa_03':
                c1, c2, c3, c4, t1,t2 = 1.26,1.69,-4.15,4.14,2.21,4.99
            elif self.output_type == 'sa_1':
                c1, c2, c3, c4, t1,t2 = 2.5,1.51,0.20,2.90,1.65,4.98
            elif  self.output_type == 'sa_3':
                c1, c2, c3, c4, t1,t2 = 3.81,1.17,1.99,3.01,0.99,4.96
            else:
                raise ValueError("Invalid input and output type combination.")
            # Direct conversion from PGA or PGV to MMI
            log_input = np.log10(input_value)
            return np.where(log_input <= t1,
                            c1 + c2 * log_input,
                            c3 + c4 * log_input)

        else:
            raise ValueError("Invalid input and output type combination.")
            
     # Implementation of the TianEtAl21 conversion logic
    def TianEtAl21(self):

        input_value = np.array(self.input_value)
            
        # MMI to PGM 
        if self.input_type == 'MMI' and self.output_type in ['PGA', 'PGV','sa_03','sa_1','sa_3']:
            
            # Define the coefficients for MMI to PGA/PGV conversion
            if self.output_type == 'PGA':
                c1, c2= 2.906,0.554
            elif self.output_type == 'PGV':  # PGV
                c1, c2 = 3.310,3.233
            elif self.output_type == 'sa_03':
                c1, c2 = 2.873,-0.327
            elif self.output_type == 'sa_1':
                c1, c2 = 3.065,0.540
            elif  self.output_type == 'sa_3':
                c1, c2 = 4.062,0.817
            else:
                raise ValueError("Invalid input and output type combination.")


            # Inverse conversion from MMI to PGA or PGV
            return 10 ** ((input_value - c2) / c1)

        # PGM to MMI 
        elif self.input_type in ['PGA', 'PGV','sa_03','sa_1','sa_3'] and self.output_type == 'MMI':
            if self.input_type == 'PGA':
                c1, c2= 2.906,0.554
            elif self.input_type == 'PGV' :  # PGV
                c1, c2 = 3.310,3.233
            elif self.output_type == 'sa_03':
                c1, c2 = 2.873,-0.327
            elif self.output_type == 'sa_1':
                c1, c2 = 3.065,0.540
            elif  self.output_type == 'sa_3':
                c1, c2 = 4.062,0.817
            else:
                raise ValueError("Invalid input and output type combination.")
            # Direct conversion from PGA or PGV to MMI
            log_input = np.log10(input_value)
            return (c1*log_input)+c2

        else:
            raise ValueError("Invalid input and output type combination.")
    
    def WuEtAl03(self):
        input_value = np.array(self.input_value)

        if self.input_type == 'PGV' and self.output_type == 'MMI':
            return 2.14 + 1.89 * np.log10(input_value)

        elif self.input_type == 'MMI' and output_type == 'PGV':
            return 10 ** ((input_value - 2.14) / 1.89)

        else:
            raise ValueError("Invalid input and output type combination.")

        # Implementation of the Atkinson conversion logic
    def Atkinsonandkaka07(self):

        input_value = np.array(self.input_value)
            
        # MMI to PGM 
        if self.input_type == 'MMI' and self.output_type in ['PGA', 'PGV','sa_03','sa_1','sa_3']:
            
            # Define the coefficients for MMI to PGA/PGV conversion
            if self.output_type == 'PGA':
                c1, c2, c3 = 2.315,1.319,0.372
            elif self.output_type == 'PGV':  # PGV
                c1, c2, c3  = 4.398,1.916,0.280
            elif self.output_type == 'sa_03':
                c1, c2, c3 = 3.567,1.596,0.255
            elif self.output_type == 'sa_1':
                c1, c2, c3 = 2.946,1.324,0.234
            elif  self.output_type == 'sa_3':
                c1, c2, c3 = 2.088,1.146,0.328
            else:
                raise ValueError("Invalid input and output type combination.")

                # Solve the quadratic equation to find LogY
            a = c3
            b = c2
            c = c1 - input_value
            discriminant = b**2 - 4*a*c

            # Consider only the positive root since LogY should be positive
            logY = (-b + np.sqrt(discriminant)) / (2 * a)
            return 10 ** logY

        # PGM to MMI 
        elif self.input_type in ['PGA', 'PGV','sa_03','sa_1','sa_3'] and self.output_type == 'MMI':
            if self.input_type == 'PGA':
                c1, c2, c3 = 2.315,1.319,0.372
            elif self.input_type == 'PGV' :  # PGV
                c1, c2, c3  = 4.398,1.916,0.280
            elif self.output_type == 'sa_03':
                c1, c2, c3 = 3.567,1.596,0.255
            elif self.output_type == 'sa_1':
                c1, c2, c3 = 2.946,1.324,0.234
            elif  self.output_type == 'sa_3':
                c1, c2, c3 = 2.088,1.146,0.328
            else:
                raise ValueError("Invalid input and output type combination.")
            # Direct conversion from PGA or PGV to MMI
            log_input = np.log10(input_value)
            return c1+c2*log_input+c3* (log_input**2)

        else:
            raise ValueError("Invalid input and output type combination.")
            
    def JMA2001(self):
        input_value = np.array(self.input_value)

        # PGA to MMI
        if self.input_type == 'PGA' and self.output_type == 'MMI':
            # Direct conversion from PGA or PGV to MMI
            log_input = np.log10(input_value)
            return 3.93*log_input-1.17
        # PGV to MMI
        if self.input_type == 'PGV' and self.output_type == 'MMI':
            # Direct conversion from PGA or PGV to MMI
            log_input = np.log10(input_value)
            return 3.93*log_input-1.17
        if self.input_type == 'sa' and self.output_type == 'MMI':
            raise ValueError("Invalid output type try PGA or PGV.")
        # MMI to PGA
        if self.input_type == 'MMI' and self.output_type == 'PGA':
            pga = (input_value +1.17)/3.94
            pga = 10 ** pga
            return pga
        # MMI to PGV
        if self.input_type == 'MMI' and self.output_type == 'PGV':
            pgv = (input_value +1.17)/3.94
            pgv = 10 ** pgv
            return pgv
        else:
            raise ValueError("Invalid input and output type combination.")

    def list_gmice(self):
        # Class description with available models and their brief descriptions
        models_descriptions = {
        'Globalgmice [MMI | PGA, PGV]': "Caprio et al. (2015) - A Global Relationship and Evaluation of Regional Dependency",
        'Bilalandaskan14 [MMI | PGA, PGV]': "Mustafa Bilal and Aysegul Askan (2014) - Relationships between Felt Intensity and Recorded Ground Motion Parameters",
        'Zaniniandhofer19 [EMS | PGA, PGV]': "Mariano Zanini and Lorenzo Hofer (2019) - Reversible ground motion-to-intensity conversion equations based on the EMS-98 scale",
        'Fami10 [MCS | PGA, PGV]': "Faenza and Michelini (2010) - Regression analysis of MCS intensity and ground motion parameters in Italy",
        'WordenEtAl12 [MMI | PGA, PGV, sa_03, sa_1, sa_3]': "Worden et al. (2012) - Probabilistic Relationships between Ground‐Motion Parameters and Modified Mercalli Intensity in California",
        'TianEtAl21 [MMI| PGA, PGV, sa_03, sa_1, sa_3]' : "Tian et al. (2021). New Ground Motion to Intensity Conversion Equations for China"
        ,'WuEtAl03 [MMI | PGV]' : "Wu et al. (2003). Relationship between Peak Ground Acceleration, Peak Ground Velocity, and Intensity in Taiwan"
        , 'Atkinsonandkaka07 [MMI | PGA, PGV, sa_03, sa_1, sa_3]': "Atkinson and Kaka (2007). Relationships between Felt Intensity and Instrumental Ground Motion in the Central United States and California"
            ,'JMA2001 [MMI, JMA | PGA, PGV]' : "Khosrow T. Shabestaria and Fumio Yamazaki (2001). A Proposal of Instrumental Seismic Intensity Scale Compatible with MMI Evaluated from Three-Component Acceleration Records"}
        print("Available GMICE Models and Descriptions:")
        for model, description in models_descriptions.items():
            print(f"\033[1m{model}\033[0m: {description}")
            print('------')
