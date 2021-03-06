import os
import sys
from nipype.interfaces.base import CommandLine
from nipype.pipeline.plugins import SLURMGraphPlugin
from nipype.pipeline.plugins.slurmgraph import logger
import numpy
from functools import reduce

BASE_COMMAND = sys.executable + ' -c '
IMPORTS = '"import sys; import multiprocessing as mp; import os; '
START_PROCESSES = 'list(map(' \
                  'lambda script_file_name: mp.Process(' \
                  "target=lambda script: os.system('" + sys.executable +" %s' % script)," \
                  "args=(script_file_name,)" \
                  ').start(),' \
                  "sys.argv[1].split(',')" \
                  '))" '
COMMAND = BASE_COMMAND + IMPORTS + START_PROCESSES

"""
Args for the command go in as a comma separated string of files to run. e.g. "some_script.py,another_script.py"
"""

"""
This is untested code to send the pipeline graph to a remote cluster using SLURM and multiprocessing.
The graph nodes are partitioned such that each set in the partition has no dependencies on its own member nodes. 
There is no constraint on how each partition set depends on nodes in other partition sets.

Partition sets are further broken up into batches which can be run in parallel on a multi-core machine using the multiprocessing library.

Each individual batch is schedule on a remote cluster using SLURM.

Below is an example of how to use this plugin to run a nipype workflow:

from enlnipypetools.plugins.slurmgraphmultiproc import SLURMGraphMultiProcPlugin
workflow.run(plugin=SLURMGraphMultiProcPlugin(plugin_args=
{
    'template': "#!/bin/bash\n#SBATCH --partition=<some-partition>\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --mem-per-cpu=500",
    'cores_per_compute_node': 16,
    'mem_per_compute_node_mb': 1000,
    'max_mem_per_task_mb': 450,
}))

"""


class SLURMGraphMultiProcPlugin(SLURMGraphPlugin):
    def __init__(self, **kwargs):
        if 'plugin_args' in kwargs and kwargs['plugin_args']:
            if 'cores_per_compute_node' in kwargs['plugin_args']:
                self._cores_per_compute_node: int = kwargs['plugin_args']['cores_per_compute_node']
            if 'mem_per_compute_node_mb' in kwargs['plugin_args']:
                self._mem_per_compute_node: int = kwargs['plugin_args']['mem_per_compute_node_mb']
            if 'max_mem_per_task_mb' in kwargs['plugin_args']:
                self._max_mem_per_task: int = kwargs['plugin_args']['max_mem_per_task_mb']
                # the maximum amount of memory any one node is expected to use.
        super().__init__(**kwargs)

    def calculate_job_partition(self, dependencies: dict):
        maximum_tasks_per_compute_node = min(self._cores_per_compute_node, int(self._mem_per_compute_node / self._max_mem_per_task))

        def remove_ids_with_dependency_within_subset(dependency_subset: dict):
            removed_dependencies = {}
            dependencies_for_subset = set([item for sublist in dependency_subset.values() for item in sublist])
            for idx in dependency_subset.keys():
                if dependencies_for_subset.__contains__(idx):
                    removed_dependencies.update({idx: dependency_subset[idx]})

            for idx in removed_dependencies.keys():
                dependency_subset.pop(idx)
            return removed_dependencies

        def recursively_resolve_dependency_crashes(subset_to_uncrash, new_subsets_list):
            subsets_to_add = remove_ids_with_dependency_within_subset(subset_to_uncrash)
            if len(subsets_to_add) > 0:
                new_subsets_list.append(subsets_to_add)
                recursively_resolve_dependency_crashes(subsets_to_add, new_subsets_list)

        new_subsets = []
        dependency_copy = dependencies.copy()
        recursively_resolve_dependency_crashes(dependency_copy, new_subsets)
        new_subsets.insert(0, dependency_copy)

        def partition_subset_by_max_jobs(subset: dict, max_jobs: int):
            subset_partition = []
            part = []
            for key in subset.keys():
                if len(part) == max_jobs:
                    subset_partition.append(part)
                    part = []
                part.append(key)
            if len(part) > 0: subset_partition.append(part)
            return subset_partition

        job_partitions = list(map(lambda subset: partition_subset_by_max_jobs(subset, maximum_tasks_per_compute_node), new_subsets))

        job_partitions_flat = []
        for sublist in job_partitions:
            for partition in sublist:
                job_partitions_flat.insert(0, partition)

        def get_dependencies_for_partition(partition: list, deps: dict, partitions: list):
            deps_for_partition = list(map(lambda idx: deps[idx], partition))
            job_deps_for_partition = list(set([item for sublist in deps_for_partition for item in sublist]))

            partition_dependencies = []
            for dep in job_deps_for_partition:
                for i in range(len(partitions)):
                    if partitions[i].__contains__(dep):
                        partition_dependencies.append(i)
                        continue
            return list(set(partition_dependencies))

        partitions_with_dependencies = {}
        for i in range(len(job_partitions_flat)):
            partitions_with_dependencies.update({i: [job_partitions_flat[i], get_dependencies_for_partition(job_partitions_flat[i], dependencies, job_partitions_flat)]})

        return partitions_with_dependencies

    def _submit_graph(self, pyfiles, dependencies, nodes):
        def make_job_name(partition_number, job_numbers_list, nodeslist):
            """
            - jobnumber: The index number of the job to create
            - nodeslist: The name of the node being processed
            - return: A string representing this job to be displayed by SLURM
            """
            max_job_idx = min(len(job_numbers_list), 5)
            job_numbers = reduce(lambda a,b: a + '_' + b, list(map(lambda jobnumber: str(jobnumber), job_numbers_list[:max_job_idx])))
            nodes = reduce(lambda a, b: a + '_' + b,
                                 list(map(lambda jobnumber: str(nodeslist[jobnumber]._id), job_numbers_list[:max_job_idx])))
            job_name = 'j{0}_{1}_{2}'.format(partition_number, job_numbers, nodes)
            # Condition job_name to be a valid bash identifier (i.e. - is invalid)
            job_name = job_name.replace('-', '_').replace('.', '_').replace(
                ':', '_')
            return job_name

        batch_dir, _ = os.path.split(pyfiles[0])
        submitjobsfile = os.path.join(batch_dir, 'submit_jobs.sh')
        partitions = self.calculate_job_partition(dependencies)

        cache_doneness_per_node = dict()
        # if self._dont_resubmit_completed_jobs:  # A future parameter for controlling this behavior could be added here
        #     for idx, pyscript in enumerate(pyfiles):
        #         node = nodes[idx]
        #         node_status_done = super.node_completed_status(node)
        #
        #         # if the node itself claims done, then check to ensure all
        #         # dependancies are also done
        #         if node_status_done and idx in dependencies:
        #             for child_idx in dependencies[idx]:
        #                 if child_idx in cache_doneness_per_node:
        #                     child_status_done = cache_doneness_per_node[
        #                         child_idx]
        #                 else:
        #                     child_status_done = super.node_completed_status(
        #                         nodes[child_idx])
        #                 node_status_done = node_status_done and child_status_done
        #
        #         cache_doneness_per_node[idx] = node_status_done

        with open(submitjobsfile, 'wt') as fp:
            fp.writelines('#!/usr/bin/env bash\n')
            fp.writelines('# Condense format attempted\n')

            for partition_idx in sorted(partitions.keys()):
                part = partitions[partition_idx]
                files_for_partition = numpy.asarray(pyfiles)[part[0]]
                file_list_string = reduce(lambda a,b: a + ',' + b, files_for_partition)
                partition_command = COMMAND + '"' + file_list_string + '"'

                batch_dir, name = os.path.split(files_for_partition[0])

                template, sbatch_args = self._get_args(
                    nodes[part[0][0]], ["template", "sbatch_args"])

                jobname = make_job_name(partition_idx, part[0], nodes)

                batchscript = '\n'.join(
                    (template, '%s' % (partition_command)))
                batchscriptfile = os.path.join(batch_dir,
                                               'batchscript_%s.sh' % jobname)

                batchscriptoutfile = batchscriptfile + '.o'
                batchscripterrfile = batchscriptfile + '.e'

                with open(batchscriptfile, 'wt') as batchfp:
                    batchfp.writelines(batchscript)
                    batchfp.close()

                deps = ''
                values = ''
                for partition_id in part[1]:
                    # Avoid dependancies of done jobs
                    if not self._dont_resubmit_completed_jobs or not cache_doneness_per_node[partition_id]:
                        values += "${{{0}}}:".format(
                            make_job_name(partition_id, partitions[partition_id][0], nodes))

                if values != '':  # i.e. if some jobs were added to dependency list
                    values = values.rstrip(':')
                    deps = '--dependency=afterok:%s' % values



                # Do not use default output locations if they are set in self._sbatch_args
                stderrFile = ''
                if self._sbatch_args.count('-e ') == 0:
                    stderrFile = '-e {errFile}'.format(
                        errFile=batchscripterrfile)
                stdoutFile = ''
                if self._sbatch_args.count('-o ') == 0:
                    stdoutFile = '-o {outFile}'.format(
                        outFile=batchscriptoutfile)
                # os.system(
                #     'sbatch {outFileOption} {errFileOption} {extraSBatchArgs} {dependantIndex} -J {jobNm} {batchscript}'.format(
                #         jobNm=jobname,
                #         outFileOption=stdoutFile,
                #         errFileOption=stderrFile,
                #         extraSBatchArgs=sbatch_args,
                #         dependantIndex=deps,
                #         batchscript=batchscriptfile))
                full_line = '{jobNm}=$(sbatch {outFileOption} {errFileOption} {extraSBatchArgs} {dependantIndex} -J {jobNm} {batchscript} | awk \'/^Submitted/ {{print $4}}\')\n'.format(
                    jobNm=jobname,
                    outFileOption=stdoutFile,
                    errFileOption=stderrFile,
                    extraSBatchArgs=sbatch_args,
                    dependantIndex=deps,
                    batchscript=batchscriptfile)
                fp.writelines(full_line)
        cmd = CommandLine(
            'bash',
            environ=dict(os.environ),
            resource_monitor=False,
            terminal_output='allatonce')
        cmd.inputs.args = '%s' % submitjobsfile
        cmd.run()
        logger.info('submitted all jobs to queue')
