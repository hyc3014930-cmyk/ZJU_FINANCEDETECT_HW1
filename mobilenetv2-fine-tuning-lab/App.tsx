
import React, { useState } from 'react';
import { Layers, Activity, BookOpen, Box, Save, RefreshCw, TrendingUp, ChevronRight, Layout, Recycle } from 'lucide-react';
import { AppTab } from './types';
import StructureVisualizer from './components/StructureVisualizer';
import TrainingSimulator from './components/TrainingSimulator';
import ConceptCard from './components/ConceptCard';
import LRVisualizer from './components/LRVisualizer';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.STRUCTURE);

  const MENU_ITEMS = [
    { 
      id: AppTab.STRUCTURE, 
      label: '0. 模型结构视窗', 
      icon: Box, 
      desc: 'Backbone & Head Visualization' 
    },
    { 
      id: AppTab.TRAINING, 
      label: '1. 训练策略模拟', 
      icon: Activity, 
      desc: 'Head-Only vs Two-Step' 
    },
    { 
      id: AppTab.LR_SCHEDULER, 
      label: '2. 学习率实验室', 
      icon: TrendingUp, 
      desc: 'Warmup & Decay Strategy' 
    },
    { 
      id: AppTab.CONCEPTS, 
      label: '3. 核心概念百科', 
      icon: BookOpen, 
      desc: 'Key Terminology' 
    },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case AppTab.STRUCTURE:
        return (
            <div className="space-y-6">
                <p className="text-slate-600 bg-indigo-50 p-4 rounded-lg border border-indigo-100 text-sm">
                    💡 <strong>实验指引：</strong> 本实验针对垃圾分类任务（4大类26小类）。点击右侧“开始推理”，观察 MindSpore 模型如何识别具体的垃圾物品（如电池、易拉罐等）。
                </p>
                <StructureVisualizer />
            </div>
        );
      case AppTab.TRAINING:
        return (
            <div className="space-y-6">
                 <p className="text-slate-600 bg-indigo-50 p-4 rounded-lg border border-indigo-100 text-sm">
                    💡 <strong>实验指引：</strong> 不同的微调策略决定了模型的天花板。尝试切换策略，观察“先冻结后解冻（两步法）”如何在后期突破精度瓶颈。
                </p>
                <TrainingSimulator />
            </div>
        );
      case AppTab.LR_SCHEDULER:
        return (
            <div className="space-y-6">
                <p className="text-slate-600 bg-indigo-50 p-4 rounded-lg border border-indigo-100 text-sm">
                    💡 <strong>实验指引：</strong> <code>build_lr</code> 函数是控制训练节奏的关键。调节 Warmup 步数和 Decay 策略，观察学习率曲线的变化规律。
                </p>
                <LRVisualizer />
            </div>
        );
      case AppTab.CONCEPTS:
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fadeIn pb-10">
            <ConceptCard 
                title="MindSpore 框架"
                icon={Layers}
                colorClass="border-blue-100 shadow-blue-100"
                description="最佳匹配 Ascend（昇腾）芯片的开源 AI 计算框架。"
                points={[
                    "支持全场景协同（端、边、云）",
                    "自动微分，高效执行",
                    "兼容 CPU/GPU/Ascend"
                ]}
            />
            <ConceptCard 
                title="Backbone (预训练)"
                icon={Box}
                colorClass="border-sky-100 shadow-sky-100"
                description="复用 ImageNet 上的 MobileNetV2，避免从零训练。"
                points={[
                    "提取通用特征（边缘、形状）",
                    "节省大量算力和时间",
                    "解决小数据集（垃圾分类）数据不足问题"
                ]}
            />
            <ConceptCard 
                title="Checkpoint (检查点)"
                icon={Save}
                colorClass="border-emerald-100 shadow-emerald-100"
                description="训练过程的“快照”，保存了模型参数和训练状态。"
                points={[
                    "训练后推理：直接加载用于预测",
                    "异常恢复：中断后恢复训练进度",
                    "迁移学习：加载预训练权重"
                ]}
            />
            <ConceptCard 
                title="Fine-tuning (微调)"
                icon={RefreshCw}
                colorClass="border-orange-100 shadow-orange-100"
                description="冻结 Backbone，仅训练 26 类垃圾分类 Head。"
                points={[
                    "冻结参数 requires_grad=False",
                    "仅更新全连接层权重",
                    "高效适配下游任务"
                ]}
            />
             <ConceptCard 
                title="Learning Rate (学习率)"
                icon={TrendingUp}
                colorClass="border-purple-100 shadow-purple-100"
                description="控制模型学习步伐的关键超参数。"
                points={[
                    "Warmup：初期低LR帮助稳定启动",
                    "Decay：Cosine衰减帮助精细收敛",
                    "防止梯度爆炸或陷入局部最优"
                ]}
            />
          </div>
        );
      
      default:
        return null;
    }
  };

  const activeItem = MENU_ITEMS.find(item => item.id === activeTab);

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden font-sans">
      {/* Left Sidebar */}
      <aside className="w-72 bg-white border-r border-slate-200 flex flex-col shadow-lg z-20 flex-shrink-0">
        {/* Sidebar Header */}
        <div className="p-6 border-b border-slate-100">
             <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-emerald-600 rounded-xl flex items-center justify-center text-white shadow-emerald-200 shadow-md">
                  <Recycle className="w-6 h-6" />
                </div>
                <div>
                    <h1 className="font-bold text-lg text-slate-800 tracking-tight leading-none">垃圾分类 Lab</h1>
                    <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">MindSpore Experiment</span>
                </div>
             </div>
        </div>
        
        {/* Navigation List */}
        <nav className="flex-1 overflow-y-auto py-6 px-4 space-y-2">
          {MENU_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full text-left flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 group relative
                ${activeTab === item.id 
                  ? 'bg-emerald-600 text-white shadow-md shadow-emerald-200' 
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 border border-transparent hover:border-slate-100'
                }
              `}
            >
              <item.icon className={`w-5 h-5 flex-shrink-0 ${activeTab === item.id ? 'text-emerald-100' : 'text-slate-400 group-hover:text-emerald-500'}`} />
              
              <div className="flex-1 min-w-0">
                  <div className={`font-medium text-sm truncate ${activeTab === item.id ? 'text-white' : 'text-slate-700'}`}>
                      {item.label}
                  </div>
                  <div className={`text-[11px] truncate mt-0.5 ${activeTab === item.id ? 'text-emerald-200' : 'text-slate-400 group-hover:text-slate-500'}`}>
                      {item.desc}
                  </div>
              </div>

              {activeTab === item.id && (
                  <ChevronRight className="w-4 h-4 text-emerald-300" />
              )}
            </button>
          ))}
        </nav>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/50">
             <div className="flex items-center gap-2 p-3 bg-white border border-slate-200 rounded-lg">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-xs font-mono text-slate-500">MindSpore Ready</span>
             </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">


        {/* Scrollable Content View */}
        <div className="flex-1 overflow-y-auto p-8 scroll-smooth bg-slate-50/80 relative">

           <div className="max-w-6xl mx-auto min-h-full">
              {/* Page Title Section */}
              <div className="mb-8 animate-fadeIn">
                 <h1 className="text-3xl font-bold text-slate-800 mb-2 flex items-center gap-3">
                    {activeItem && <activeItem.icon className="w-8 h-8 text-emerald-600" />}
                    {activeItem?.label.split(' ')[1]}
                 </h1>
                  <p className="text-slate-500 max-w-3xl leading-relaxed">
                    {activeTab === AppTab.STRUCTURE && "深入理解 MindSpore 框架下的 MobileNetV2 结构。通过可视化交互，观察模型如何从图像中提取特征，并识别 26 类生活垃圾。"}
                    {activeTab === AppTab.TRAINING && "对比 Head-Only 与 Fine-Tuning 策略的效果。理解为什么在小数据集（垃圾分类）上，先冻结 Backbone 是最高效的策略。"}
                    {activeTab === AppTab.LR_SCHEDULER && "动态运行 build_lr 函数。调节 Warmup 步数和 Decay 策略，直观感受学习率如何随训练步数动态调整。"}
                    {activeTab === AppTab.CONCEPTS && "掌握 MindSpore 实验中的核心概念：从 Checkpoint 机制到预训练模型的迁移学习原理。"}
                  </p>
              </div>

              {/* Dynamic Component */}
              <div className="animate-fadeIn">
                {renderContent()}
              </div>
           </div>
           
           {/* Bottom Spacer */}
           <div className="h-12" /> 
        </div>
      </main>
    </div>
  );
};

export default App;
