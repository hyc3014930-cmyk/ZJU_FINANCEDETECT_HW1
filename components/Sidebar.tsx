import React from 'react';
import { 
  LayoutDashboard, 
  BrainCircuit, 
  Layers, 
  Ruler, 
  Beaker 
} from 'lucide-react';
import { StepType } from '../types';

interface SidebarProps {
  currentStep: StepType;
  onStepChange: (step: StepType) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentStep, onStepChange }) => {
  const menuItems = [
    { id: StepType.OVERVIEW, icon: LayoutDashboard, label: '0. 核心流程视窗', subLabel: 'Pipeline Overview' },
    { id: StepType.LLM_INFERENCE, icon: BrainCircuit, label: '1. Prompt 策略模拟', subLabel: 'LLM Inference' },
    { id: StepType.BATCH_PROCESSING, icon: Layers, label: '2. 批量处理逻辑', subLabel: 'Batch Processing' },
    { id: StepType.EMBEDDING_METRICS, icon: Ruler, label: '3. 核心概念百科', subLabel: 'Scoring Metrics' },
  ];

  return (
    <div className="w-72 bg-slate-900 h-screen flex flex-col border-r border-slate-800 text-white shrink-0">
      <div className="p-6 border-b border-slate-800">
        <div className="flex items-center space-x-2 text-teal-400 mb-1">
          <Beaker className="w-6 h-6" />
          <span className="font-bold text-lg tracking-tight">TCM AI Lab</span>
        </div>
        <h1 className="text-xl font-bold mt-4 text-white">中医辨证 Lab</h1>
        <p className="text-xs text-slate-400 mt-1 uppercase tracking-wider">Experiment Environment</p>
      </div>

      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {menuItems.map((item) => {
          const isActive = currentStep === item.id;
          const Icon = item.icon;
          
          return (
            <button
              key={item.id}
              onClick={() => onStepChange(item.id)}
              className={`w-full flex items-center p-3 rounded-xl transition-all duration-200 group text-left ${
                isActive 
                  ? 'bg-gradient-to-r from-teal-600 to-teal-700 text-white shadow-lg shadow-teal-900/50' 
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <div className={`p-2 rounded-lg mr-3 ${isActive ? 'bg-white/20' : 'bg-slate-800 group-hover:bg-slate-700'}`}>
                <Icon className="w-5 h-5" />
              </div>
              <div>
                <div className="font-medium text-sm">{item.label}</div>
                <div className={`text-xs ${isActive ? 'text-teal-100' : 'text-slate-500'}`}>{item.subLabel}</div>
              </div>
            </button>
          );
        })}
      </nav>

      <div className="p-4 bg-slate-950 border-t border-slate-800 text-xs text-slate-500">
        <p>Environment: Demo / Python 3.9</p>
        <p className="mt-1">Framework: OpenAI / Sklearn</p>
      </div>
    </div>
  );
};
