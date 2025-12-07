import React from 'react';

const StartPage: React.FC<{ onSelect: (m: number) => void; onCancel?: () => void }> = ({ onSelect, onCancel }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-6">
      <div className="max-w-5xl w-full">
        <div className="bg-white p-8 rounded-2xl shadow-2xl border border-slate-200 relative">
          {/* 面板右上角背景图片 */}
          <div className="absolute top-4 right-12 w-20 h-20 bg-cover bg-center opacity-25" style={{ 
            backgroundImage: 'url(/WPS图片.png)',
            filter: 'hue-rotate(200deg) saturate(2.5) brightness(1.2)'
          }}></div>
          
          <h1 className="text-2xl font-bold text-slate-800 mb-2">请选择一个作业模块</h1>
          <p className="text-sm text-slate-500 mb-6">下面列出三个模块，点击进入相应实验与交互演示。</p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <button onClick={() => onSelect(1)} className="text-left bg-blue-50 border border-blue-100 rounded-xl p-5 hover:shadow-lg transition-all hover:-translate-y-1">
              <div className="text-base text-blue-700 font-bold mb-2">金融异常检测</div>
              <div className="text-xs text-slate-500 mb-3">核心知识点：</div>
              <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
                <li>时间序列分析</li>
                <li>异常值检测算法</li>
                <li>统计建模</li>
              </ul>
            </button>

            <button onClick={() => onSelect(2)} className="text-left bg-emerald-50 border border-emerald-100 rounded-xl p-5 hover:shadow-lg transition-all hover:-translate-y-1">
              <div className="text-base text-emerald-700 font-bold mb-2">垃圾分类视觉模型</div>
              <div className="text-xs text-slate-500 mb-3">核心知识点：</div>
              <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
                <li>MobileNetV2 架构</li>
                <li>迁移学习</li>
                <li>图像分类</li>
              </ul>
            </button>

            <button onClick={() => onSelect(3)} className="text-left bg-purple-50 border border-purple-100 rounded-xl p-5 hover:shadow-lg transition-all hover:-translate-y-1">
              <div className="text-base text-purple-700 font-bold mb-2">中医辨证模型</div>
              <div className="text-xs text-slate-500 mb-3">核心知识点：</div>
              <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
                <li>LLM 推理</li>
                <li>Prompt 工程</li>
                <li>批量处理</li>
              </ul>
            </button>
          </div>

          <div className="mt-6 flex justify-end gap-2">
            <button onClick={() => onCancel && onCancel()} className="px-4 py-2 rounded bg-slate-100 text-slate-700 text-sm hover:bg-slate-200 transition">跳过</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StartPage;
