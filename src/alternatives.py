import dataclasses
from abc import ABC, abstractmethod
import pandas as pd



class AlternativesProcessor(ABC):
    def __init__(self, explanation: str = ""):
        self.explanation = explanation
    @abstractmethod
    def process(self, df: pd.DataFrame)->pd.DataFrame:
        pass

@dataclasses.dataclass
class Alternative:
    explanation: str
    processor: AlternativesProcessor

class AlternativesProcessorPipeline(AlternativesProcessor):
    def __init__(self, alternatives: list[Alternative]):
        super().__init__()
        self.alternatives = alternatives

    def process(self, df: pd.DataFrame)->pd.DataFrame:
        df['alternatives_explanation'] = ''
        for alt in self.alternatives:
            df = alt.processor.process(df)
        return df

class DistanceProcessor(AlternativesProcessor):
    def __init__(self, params, explanation):
        super().__init__(explanation)
        self.params = params

    def process(self, df)->pd.DataFrame:
        for param_set in self.params:
            df.loc[param_set['index'], 'SHOT_DISTANCE'] = param_set['distance']
            df.loc[param_set['index'], 'alternatives_explanation'] = self.explanation
            # Adjust points after distance
            if param_set['distance'] >= 23:
                df.loc[param_set['index'], 'points'] = 3
            else:
                df.loc[param_set['index'], 'points'] = 2
        return df





class AlternativesCalculator(ABC):
    """
    calculate alternatives based on naive understanding of the game
    """
    @abstractmethod
    def calculate_alternatives(self, df: pd.DataFrame)->list[Alternative]:
        pass

class ThreePointAlternativesCalculator(AlternativesCalculator):
    """
    check if throw could have been three points
    """
    def __init__(self, min_distance = 21):
        self.min_distance = min_distance
    def calculate_alternatives(self, df: pd.DataFrame)->list[Alternative]:
        alternatives = []
        text = "Step behind 3pt line to earn more points"
        params = []
        for i, row in df.iterrows():
            if row['SHOT_DISTANCE'] < 23 and row['SHOT_DISTANCE'] > self.min_distance:
                params.append({'index': i, 'distance': 23})
        processor = DistanceProcessor(params, text)
        print("len params", len(params))
        alternatives.append(Alternative(text, processor))
        return alternatives

class AlternativesCalculatorPipeline(AlternativesCalculator):
    """
    pipeline for alternatives calculating
    """
    def __init__(self, individual_calculators):
        self.individual_calculators = individual_calculators

    def calculate_alternatives(self, df: pd.DataFrame)->list[Alternative]:
        alternatives = []
        for calc in self.individual_calculators:
            alternatives += calc.calculate_alternatives(df)
        return alternatives