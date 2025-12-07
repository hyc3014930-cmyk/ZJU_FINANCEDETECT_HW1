
export enum TrainingStrategy {
  HEAD_ONLY = 'HEAD_ONLY',
  TWO_STEP = 'TWO_STEP'
}

export interface ChartDataPoint {
  epoch: number;
  accuracy: number;
  phase: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
}

export enum AppTab {
  STRUCTURE = 'STRUCTURE',
  TRAINING = 'TRAINING',
  LR_SCHEDULER = 'LR_SCHEDULER',
  CONCEPTS = 'CONCEPTS'
}

export interface CodeLine {
  num: number;
  code: string;
  indent: number;
  highlight?: boolean;
}
