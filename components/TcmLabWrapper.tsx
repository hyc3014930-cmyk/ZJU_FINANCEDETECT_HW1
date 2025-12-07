import React, { Suspense } from 'react';

// Lazy-load the TCM lab module
const LazyTcmApp = React.lazy(() => import('../TCM/App'));

const TcmLabWrapper: React.FC = () => {
  return (
    <div className="h-full w-full">
      <Suspense fallback={<div className="p-6 text-center text-slate-500">正在加载中医辨证 Lab...</div>}>
        <LazyTcmApp />
      </Suspense>
    </div>
  );
};

export default TcmLabWrapper;