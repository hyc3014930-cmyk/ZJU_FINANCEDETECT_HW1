export enum StepType {
  OVERVIEW = 'OVERVIEW',
  LLM_INFERENCE = 'LLM_INFERENCE',
  BATCH_PROCESSING = 'BATCH_PROCESSING',
  EMBEDDING_METRICS = 'EMBEDDING_METRICS'
}

export interface CodeSection {
  id: string;
  title: string;
  code: string;
  description: string;
  highlightLines?: number[];
}

export interface SimulationResult {
  syndrome: string; // 证型
  treatment: string; // 治法
  rawJson?: string;
}

export interface VectorPoint {
  x: number;
  y: number;
  label: string;
  category: 'prediction' | 'truth';
}
