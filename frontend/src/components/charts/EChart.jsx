import React from 'react';
import ReactEChartsCore from 'echarts-for-react/esm/core.js';
import * as echarts from 'echarts/core';
import { BarChart, PieChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([BarChart, PieChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

export default function EChart({ option, height = 340 }) {
  return (
    <ReactEChartsCore echarts={echarts} option={option} style={{ height }} notMerge lazyUpdate />
  );
}
