import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { InferenceLab } from './components/labs/InferenceLab';
import { BatchLab } from './components/labs/BatchLab';
import { EmbeddingLab } from './components/labs/EmbeddingLab';
import { CODE_SNIPPETS } from './constants';
import { StepType } from './types';
import { FileText, ArrowRightCircle } from 'lucide-react';

// Use the CodeViewer from the main project to avoid dependency issues
import { CodeViewer } from '../components/CodeViewer';

const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<StepType>(StepType.LLM_INFERENCE);

  const renderLab = () => {
    switch (currentStep) {
      case StepType.LLM_INFERENCE:
        return <InferenceLab />;
      case StepType.BATCH_PROCESSING:
        return <BatchLab />;
      case StepType.EMBEDDING_METRICS:
        return <EmbeddingLab />;
      case StepType.OVERVIEW:
      default:
        return (
          <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-4 p-8 text-center bg-white rounded-xl border border-slate-200 shadow-sm">
            <div className="w-24 h-24 bg-teal-50 rounded-full flex items-center justify-center mb-4">
                <FileText className="w-10 h-10 text-teal-600" />
            </div>
            <h2 className="text-2xl font-bold text-slate-800">欢迎来到中医辨证算法实验室</h2>
            <p className="max-w-md leading-relaxed">
              本项目演示了如何结合 <span className="text-teal-600 font-bold">LLM (大语言模型)</span> 与 <span className="text-indigo-600 font-bold">Embeddings (向量技术)</span> 来构建一个医疗诊断辅助系统。
            </p>
            <button 
                onClick={() => setCurrentStep(StepType.LLM_INFERENCE)}
                className="mt-4 flex items-center px-6 py-3 bg-slate-900 text-white rounded-full hover:bg-slate-800 transition-colors"
            >
                开始学习之旅 <ArrowRightCircle className="ml-2 w-5 h-5" />
            </button>
          </div>
        );
    }
  };

  const currentSnippet = CODE_SNIPPETS[currentStep];

  return (
    <div className="flex h-screen bg-[#f8fafc] text-slate-900 overflow-hidden font-sans">
      {/* Sidebar Navigation */}
      <Sidebar currentStep={currentStep} onStepChange={setCurrentStep} />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">


        {/* Content Split View */}
        <div className="flex-1 p-6 overflow-hidden flex flex-col lg:flex-row gap-6 relative">

          
          {/* Left: Code Viewer */}
          <div className="lg:w-5/12 h-1/2 lg:h-full flex flex-col min-h-0">
             <div className="mb-2 flex items-center justify-between">
                <h3 className="font-bold text-slate-700 flex items-center">
                    <FileText className="w-4 h-4 mr-2" /> 核心代码
                </h3>
             </div>
             <div className="flex-1 min-h-0 overflow-y-auto">
                <CodeViewer 
                    code={currentSnippet.code} 
                    title={`${currentSnippet.id}.py`} 
                />
             </div>
             <div className="mt-4 bg-white p-4 rounded-lg border border-slate-200 text-sm shadow-sm shrink-0">
                <h4 className="font-bold text-slate-800 mb-1">代码解析</h4>
                <p className="text-slate-600 leading-relaxed text-xs lg:text-sm">
                    {currentSnippet.description}
                </p>
             </div>
          </div>

          {/* Right: Interactive Lab */}
          <div className="lg:w-7/12 h-1/2 lg:h-full flex flex-col min-h-0">
            <div className="mb-2 flex items-center justify-between">
                <h3 className="font-bold text-slate-700 flex items-center">
                   <span className="w-2 h-2 rounded-full bg-teal-500 mr-2 animate-pulse"></span>
                   交互实验室 (Interactive Lab)
                </h3>
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto">
                {renderLab()}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};

export default App;
