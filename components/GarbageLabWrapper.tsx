import React, { Suspense } from 'react';

// Lazy-load the attached lab so the dev server doesn't try to resolve
// its optional dependencies until the module is actually selected.
const LazyLabApp = React.lazy(() => import('../mobilenetv2-fine-tuning-lab/App'));

const GarbageLabWrapper: React.FC = () => {
  return (
    <div className="h-full w-full">
      <Suspense fallback={<div className="p-6 text-center text-slate-500">正在加载垃圾分类 Lab...</div>}>
        <LazyLabApp />
      </Suspense>
    </div>
  );
};

export default GarbageLabWrapper;
