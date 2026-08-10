import React from 'react';
import ReactEChartsCore from 'echarts-for-react/esm/core.js';
import * as echarts from 'echarts/core';
import { PieChart } from 'echarts/charts';
import { LegendComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([PieChart, LegendComponent, TooltipComponent, CanvasRenderer]);

export default function PieEChart({ option, height = 340 }) {
  return (
    <ReactEChartsCore echarts={echarts} option={option} style={{ height }} notMerge lazyUpdate />
  );
}
