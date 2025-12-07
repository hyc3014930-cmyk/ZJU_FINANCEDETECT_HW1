import React, { useState } from 'react';
import { Database, Brain, Activity, Search, Code, Layers, BookOpen, MessageSquare, Scale, Image as ImageIcon, Box, LayoutGrid, Zap, ScanLine, ListVideo, Calculator, ChevronRight } from 'lucide-react';
import { TabView, Project } from './types';
import { DataView } from './components/DataView';
import { ModelView } from './components/ModelView';
import { TrainingView } from './components/TrainingView';
import { InferenceView } from './components/InferenceView';
import { IntroView } from './components/IntroView';
import StartPage from './components/StartPage';
import GarbageLabWrapper from './components/GarbageLabWrapper';
import TcmLabWrapper from './components/TcmLabWrapper';

const App: React.FC = () => {
  const [currentProject, setCurrentProject] = useState<Project>(Project.DGRAPH);
  const [activeTab, setActiveTab] = useState<string>(TabView.INTRO);
    const [showStart, setShowStart] = useState<boolean>(true);

  // Switch project handler
  const handleProjectSwitch = (p: Project) => {
    setCurrentProject(p);
    // Set default tab for each project
    if (p === Project.DGRAPH) setActiveTab(TabView.INTRO);
    if (p === Project.TCM) setActiveTab(TabView.TCM_INTRO);
    if (p === Project.VISION) setActiveTab(TabView.CV_INTRO);
  };

  const renderContent = () => {
        if (showStart) {
            return (
                <StartPage onSelect={(m) => {
                    // 1 -> 金融异常检测, 2 -> Garbage 垃圾分类视觉模型 (attached lab), 3 -> TCM 中医辨证模型
                    setShowStart(false);
                    if (m === 1) {
                        setCurrentProject(Project.DGRAPH);
                        setActiveTab(TabView.INTRO);
                    }
                    if (m === 2) {
                        setCurrentProject(Project.VISION);
                        setActiveTab(TabView.CV_INTRO);
                    }
                    if (m === 3) {
                        setCurrentProject(Project.TCM);
                        setActiveTab(TabView.TCM_INTRO);
                    }
                }} onCancel={() => setShowStart(false)} />
            );
        }
    // 1. DGraph (Finance)
    if (currentProject === Project.DGRAPH) {
        switch (activeTab) {
        case TabView.INTRO: return <IntroView />;
        case TabView.DATA: return <DataView />;
        case TabView.MODEL: return <ModelView />;
        case TabView.TRAINING: return <TrainingView />;
        case TabView.INFERENCE: return <InferenceView />;
        default: return <IntroView />;
        }
    }
    // Render Vision (Garbage Lab) when selected
    if (currentProject === Project.VISION) {
        return <GarbageLabWrapper />;
    }
    // Render TCM (Traditional Chinese Medicine) Lab when selected
    if (currentProject === Project.TCM) {
        return <TcmLabWrapper />;
    }

    return null;
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 flex font-sans overflow-hidden">
      
      {/* 1. Global Project Sidebar (Leftmost) */}
      <aside className="w-20 bg-slate-900 flex flex-col items-center py-6 gap-6 z-20 shadow-xl flex-shrink-0">
         <div className="bg-blue-600 p-3 rounded-xl text-white shadow-lg shadow-blue-900/50 mb-4 w-10 h-10 flex items-center justify-center">
             <span className="font-bold text-sm">ZJU</span>
         </div>
         
                <ProjectIcon 
                    active={currentProject === Project.DGRAPH} 
                    onClick={() => handleProjectSwitch(Project.DGRAPH)}
                    icon={<LayoutGrid size={24} />}
                    label="金融异常检测"
                    color="text-blue-400"
                />
                <ProjectIcon 
                    active={currentProject === Project.VISION} 
                    onClick={() => handleProjectSwitch(Project.VISION)}
                    icon={<ImageIcon size={24} />}
                    label="垃圾分类"
                    color="text-emerald-400"
                />
                <ProjectIcon 
                    active={currentProject === Project.TCM} 
                    onClick={() => handleProjectSwitch(Project.TCM)}
                    icon={<Brain size={24} />}
                    label="中医辨证"
                    color="text-purple-400"
                />
      </aside>

      {/* 2. Project Navigation Sidebar & Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        
        {/* Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 flex-shrink-0">
            <div>
                <h1 className="text-lg font-bold text-slate-800 tracking-tight flex items-center gap-2">
                    {currentProject === Project.DGRAPH && <><LayoutGrid size={20} className="text-blue-600"/> DGraphFin 金融风控模型</>}
                    {currentProject === Project.TCM && <><Brain size={20} className="text-emerald-600"/> TCM 中医大模型助手</>}
                    {currentProject === Project.VISION && <><Box size={20} className="text-cyan-600"/> Garbage 垃圾分类视觉模型</>}
                </h1>
            </div>
            <div className="flex items-center gap-2 text-xs font-bold text-slate-500 bg-slate-100 px-3 py-1.5 rounded-full">
                <Activity size={14} />
                <span>AI Teaching Platform v2.1</span>
            </div>
        </header>

        {/* Workspace */}
        <div className="flex-1 flex overflow-hidden">
            
            {/* Sub-Navigation Sidebar (hidden when Vision or TCM project selected so embedded lab shows its own menu) */}
            {currentProject !== Project.VISION && currentProject !== Project.TCM && (
            <aside className="w-72 bg-white border-r border-slate-200 flex flex-col shadow-lg z-20 flex-shrink-0">
                {/* Sidebar Header */}
                <div className="p-6 border-b border-slate-100">
                      <div className="flex items-center gap-3">
                         <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white shadow-blue-200 shadow-md">
                           <Box className="w-6 h-6" />
                         </div>
                        <div>
                            <h1 className="font-bold text-lg text-slate-800 tracking-tight leading-none">金融异常检测 Lab</h1>
                            <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">Finance Experiment</span>
                        </div>
                     </div>
                </div>
                
                {/* Navigation List */}
                <nav className="flex-1 overflow-y-auto py-6 px-4 space-y-2">
                
                {currentProject === Project.DGRAPH && (
                    <>
                        <NavButton active={activeTab === TabView.INTRO} onClick={() => setActiveTab(TabView.INTRO)} icon={<BookOpen size={18} />} label="0. 项目背景" />
                        <NavButton active={activeTab === TabView.DATA} onClick={() => setActiveTab(TabView.DATA)} icon={<Database size={18} />} label="1. 数据准备" />
                        <NavButton active={activeTab === TabView.MODEL} onClick={() => setActiveTab(TabView.MODEL)} icon={<Layers size={18} />} label="2. MLP 模型架构" />
                        <NavButton active={activeTab === TabView.TRAINING} onClick={() => setActiveTab(TabView.TRAINING)} icon={<Activity size={18} />} label="3. 训练循环" />
                        <NavButton active={activeTab === TabView.INFERENCE} onClick={() => setActiveTab(TabView.INFERENCE)} icon={<Search size={18} />} label="4. 预测推理" />
                    </>
                )}
                </nav>
                
                {/* Sidebar Footer */}
                <div className="p-4 border-t border-slate-100 bg-slate-50/50">
                     <div className="flex items-center gap-2 p-3 bg-white border border-slate-200 rounded-lg">
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                        <span className="text-xs font-mono text-slate-500">Python 3.9 Ready</span>
                     </div>
                </div>
            </aside>
            )}

            {/* Viewport */}
            <main className="flex-1 bg-slate-50 relative overflow-hidden flex flex-col">
                <div className="flex-1 overflow-y-auto">
                    {renderContent()}
                </div>
            </main>

        </div>
      </div>
    </div>
  );
};

// --- Sub Components ---

const ProjectIcon: React.FC<{ active: boolean, onClick: () => void, icon: React.ReactNode, label: string, color: string }> = ({ active, onClick, icon, label, color }) => (
    <button 
       onClick={onClick}
       className={`w-12 h-12 rounded-2xl flex flex-col items-center justify-center gap-1 transition-all duration-300 group relative ${active ? 'bg-slate-800' : 'hover:bg-slate-800/50'}`}
    >
        <div className={`transition-colors ${active ? color : 'text-slate-500 group-hover:text-slate-300'}`}>
            {icon}
        </div>
        <span className={`text-[9px] font-bold ${active ? 'text-white' : 'text-slate-500'}`}>{label}</span>
        {active && <div className={`absolute left-0 w-1 h-8 rounded-r-full ${color.replace('text', 'bg')}`}></div>}
    </button>
);

const NavButton: React.FC<{ active: boolean, onClick: () => void, icon: React.ReactNode, label: string }> = ({ active, onClick, icon, label }) => {
  // 添加子标签，从label中提取数字部分作为子标签
  const subLabels = {
    '0. 项目背景': 'Project Context',
    '1. 数据准备': 'Data Preparation',
    '2. MLP 模型架构': 'Model Architecture',
    '3. 训练循环': 'Training Loop',
    '4. 预测推理': 'Inference'
  };
  
  return (
    <button
      onClick={onClick}
      className={`w-full text-left flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 group relative
        ${active 
          ? 'bg-blue-600 text-white shadow-md shadow-blue-200' 
          : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 border border-transparent hover:border-slate-100'
        }
      `}
    >
      <div className={`w-5 h-5 flex-shrink-0 ${active ? 'text-blue-100' : 'text-slate-400 group-hover:text-blue-500'}`}>{icon}</div>
      
      <div className="flex-1 min-w-0">
          <div className={`font-medium text-sm truncate ${active ? 'text-white' : 'text-slate-700'}`}>
              {label}
          </div>
          <div className={`text-[11px] truncate mt-0.5 ${active ? 'text-blue-200' : 'text-slate-400 group-hover:text-slate-500'}`}>
              {subLabels[label as keyof typeof subLabels] || ''}
          </div>
      </div>

      {active && (
          <ChevronRight className="w-4 h-4 text-blue-300" />
      )}
    </button>
  );
};

export default App;