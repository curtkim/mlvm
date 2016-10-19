# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|

  config.vm.provider "virtualbox" do |v|
    v.memory = 2048
    v.cpus = 4
  end

  config.vm.box = "ubuntu/trusty64"

  config.vm.provision :ansible do |ansible|
    #ansible.verbose = 'vvv'
    ansible.playbook = 'site.yml'
    ansible.extra_vars = {}
    ansible.sudo = true
  end

  config.vm.define "sandbox" do |box|
    box.vm.hostname = "sandbox"
  end

end